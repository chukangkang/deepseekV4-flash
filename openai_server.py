import asyncio
import concurrent.futures
import json
import os
import queue
import sys
import threading
import time
import uuid
from argparse import ArgumentParser
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

def _log(msg):
    print(f"[openai_server] {msg}", flush=True)

def _has_uvloop():
    try:
        import uvloop  # noqa: F401
        return True
    except ImportError:
        return False

def _has_httptools():
    try:
        import httptools  # noqa: F401
        return True
    except ImportError:
        return False

import torch
import torch.distributed as dist
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

from generate_pp import TP_SIZE, load_pp_weights, pp_forward, pp_next_token
from model import ModelArgs, Transformer

current_dir = os.path.dirname(os.path.abspath(__file__))
encoding_dir = os.path.join(current_dir, "encoding")
sys.path.insert(0, os.path.abspath(encoding_dir))
from encoding_dsv4 import encode_messages, parse_message_from_completion_text


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    max_tokens: int = 256
    temperature: float = 0.6
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    reasoning_effort: Optional[str] = None


class RequestHandle:
    def __init__(self, streaming: bool = False):
        self.event = threading.Event()
        self.result = None
        self.error = None
        self.streaming = streaming
        # For true streaming: scheduler pushes (type, data) tuples
        # Types: ("token", str), ("finish", {finish_reason, usage}), ("error", Exception)
        self.token_queue: Optional[queue.Queue] = queue.Queue() if streaming else None

    def set_result(self, result: Dict[str, Any]):
        self.result = result
        self.event.set()

    def set_error(self, error: Exception):
        self.error = error
        self.event.set()
        if self.token_queue is not None:
            self.token_queue.put(("error", error))

    def push_token(self, text: str):
        if self.token_queue is not None and text:
            self.token_queue.put(("token", text))

    def push_finish(self, finish_reason: str, usage: Dict[str, Any]):
        if self.token_queue is not None:
            self.token_queue.put(("finish", {"finish_reason": finish_reason, "usage": usage}))


@dataclass
class PendingRequest:
    request_id: str
    body: ChatCompletionRequest
    created: int
    handle: RequestHandle


@dataclass
class PreparedRequest:
    pending: PendingRequest
    messages: List[Dict[str, Any]]
    prompt_tokens: List[int]
    prompt_key: str
    temperature: float
    top_p: float
    max_new_tokens: int
    stop: Optional[Union[str, List[str]]]


@dataclass
class StreamContext:
    """Tracks per-request incremental decoding state for true streaming."""
    handle: RequestHandle
    request_id: str
    created: int
    stop: Optional[Union[str, List[str]]]
    prompt_len: int
    gen_tokens: List[int] = field(default_factory=list)
    prev_text: str = ""
    finished: bool = False


class BatchedOpenAIServer:
    def __init__(
        self,
        model: Transformer,
        tokenizer,
        ctrl_group,
        global_rank: int,
        pp_rank: int,
        pp_peer_rank: int,
        hc_mult: int,
        dim: int,
        vocab_size: int,
        max_batch_size: int,
        batch_timeout_ms: int,
        prefill_chunk_size: int,
        max_batch_total_tokens: int,
        release_kv_after_batch: bool,
        model_name: str,
        max_queue_size: int,
        request_timeout_s: int = 120,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.ctrl_group = ctrl_group
        self.global_rank = global_rank
        self.pp_rank = pp_rank
        self.pp_peer_rank = pp_peer_rank
        self.hc_mult = hc_mult
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.prefill_chunk_size = prefill_chunk_size
        self.max_batch_total_tokens = max_batch_total_tokens
        self.release_kv_after_batch = release_kv_after_batch
        self.model_name = model_name
        self.max_queue_size = max_queue_size
        self.request_timeout_s = request_timeout_s
        self.pending: List[PendingRequest] = []
        self.cv = threading.Condition()
        self.stopping = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.prefix_cache: Dict[str, List[int]] = {}

    def start(self):
        if self.global_rank == 0:
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.scheduler_thread.start()

    def shutdown(self):
        self.stopping = True
        with self.cv:
            self.cv.notify_all()
        if self.scheduler_thread is not None:
            self.scheduler_thread.join(timeout=5)

    def submit(self, body: ChatCompletionRequest,
               request_id: Optional[str] = None,
               created: Optional[int] = None) -> RequestHandle:
        handle = RequestHandle(streaming=body.stream)
        pending = PendingRequest(
            request_id=request_id or f"chatcmpl-{uuid.uuid4().hex}",
            body=body,
            created=created or int(time.time()),
            handle=handle,
        )
        with self.cv:
            if len(self.pending) >= self.max_queue_size:
                raise HTTPException(status_code=503, detail="request queue is full")
            self.pending.append(pending)
            self.cv.notify()
        return handle

    def worker_loop(self):
        while True:
            payloads = [None]
            dist.broadcast_object_list(payloads, src=0, group=self.ctrl_group)
            payload = payloads[0]
            if payload["type"] == "shutdown":
                break
            self._distributed_generate(payload)

    def _scheduler_loop(self):
        torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
        torch.set_default_device("cuda")
        while not self.stopping:
            batch = self._dequeue_batch()
            if not batch:
                continue
            _log(f"dequeued batch of {len(batch)} requests, pending remaining: {len(self.pending)}")
            self._process_batch(batch)

    def _dequeue_batch(self) -> List[PendingRequest]:
        with self.cv:
            while not self.pending and not self.stopping:
                self.cv.wait(timeout=0.1)
            if self.stopping:
                return []
            # Drain all currently available requests
            batch: List[PendingRequest] = []
            while self.pending and len(batch) < self.max_batch_size:
                batch.append(self.pending.pop(0))
            # Wait up to batch_timeout_ms for more requests to accumulate.
            # After each new arrival, wait an additional short settle period
            # so that a burst of HTTP requests all land in the same batch.
            deadline = time.time() + self.batch_timeout_ms / 1000.0
            settle_ms = 0.1  # 100ms settle after each new arrival
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                wait_time = min(remaining, settle_ms)
                if not self.pending:
                    self.cv.wait(timeout=wait_time)
                if self.pending:
                    while self.pending and len(batch) < self.max_batch_size:
                        batch.append(self.pending.pop(0))
                    # Reset settle: wait a bit more for stragglers
                    deadline = min(deadline, time.time() + settle_ms)
                else:
                    # Nothing arrived during wait — if we already have items, stop waiting
                    if batch:
                        break
            return batch

    def _normalize_messages(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        response_format: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for msg in messages:
            item = dict(msg)
            content = item.get("content")
            if isinstance(content, list):
                texts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(block.get("text", ""))
                item["content"] = "\n".join(texts)
            normalized.append(item)
        if tools or response_format:
            attached = False
            for msg in normalized:
                if msg.get("role") in {"system", "developer", "user"}:
                    if tools:
                        msg["tools"] = tools
                    if response_format:
                        msg["response_format"] = response_format
                    attached = True
                    break
            if not attached:
                item: Dict[str, Any] = {"role": "system", "content": ""}
                if tools:
                    item["tools"] = tools
                if response_format:
                    item["response_format"] = response_format
                normalized.insert(0, item)
        return normalized

    def _prepare_request(self, pending: PendingRequest) -> PreparedRequest:
        body = pending.body
        if body.n != 1:
            raise ValueError("only n=1 is supported")
        if body.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        messages = self._normalize_messages(body.messages, body.tools, body.response_format)
        prompt = encode_messages(messages, thinking_mode="chat", reasoning_effort=body.reasoning_effort)
        prompt_key = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        prompt_tokens = self.prefix_cache.get(prompt_key)
        if prompt_tokens is None:
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            self.prefix_cache[prompt_key] = prompt_tokens
        if len(prompt_tokens) > self.model.max_seq_len:
            raise ValueError(f"prompt length {len(prompt_tokens)} exceeds max_seq_len {self.model.max_seq_len}")
        return PreparedRequest(
            pending=pending,
            messages=messages,
            prompt_tokens=prompt_tokens,
            prompt_key=prompt_key,
            temperature=body.temperature,
            top_p=body.top_p,
            max_new_tokens=body.max_tokens,
            stop=body.stop,
        )

    def _process_batch(self, batch: List[PendingRequest]):
        # Drop requests that have been waiting too long (client likely timed out)
        now = int(time.time())
        live_batch: List[PendingRequest] = []
        for pending in batch:
            if now - pending.created > self.request_timeout_s:
                pending.handle.set_error(TimeoutError(f"request waited {now - pending.created}s in queue, dropped"))
            else:
                live_batch.append(pending)
        prepared: List[PreparedRequest] = []
        for pending in live_batch:
            try:
                prepared.append(self._prepare_request(pending))
            except Exception as exc:
                pending.handle.set_error(exc)
        if not prepared:
            return
        subbatches = self._split_prepared_batches(prepared)
        dropped = len(batch) - len(live_batch)
        if dropped > 0 or len(subbatches) > 1:
            _log(f"batch={len(batch)} → live={len(live_batch)} (dropped={dropped}), split into {len(subbatches)} sub-batches of sizes {[len(sb) for sb in subbatches]}")
        pending_subbatches = list(subbatches)
        while pending_subbatches:
            subbatch = pending_subbatches.pop(0)
            stream_ctxs: List[Optional[StreamContext]] = []
            for item in subbatch:
                # Check for carried-forward stream context from V2 continuation
                cont_sctx = self._cont_stream_ctxs.pop(id(item), None) if hasattr(self, '_cont_stream_ctxs') else None
                if cont_sctx is not None:
                    stream_ctxs.append(cont_sctx)
                elif item.pending.handle.streaming:
                    stream_ctxs.append(StreamContext(
                        handle=item.pending.handle,
                        request_id=item.pending.request_id,
                        created=item.pending.created,
                        stop=item.stop,
                        prompt_len=len(item.prompt_tokens),
                    ))
                else:
                    stream_ctxs.append(None)
            payload = {
                "type": "generate",
                "prompt_tokens": [item.prompt_tokens for item in subbatch],
                "temperatures": [item.temperature for item in subbatch],
                "top_ps": [item.top_p for item in subbatch],
                "max_new_tokens": [item.max_new_tokens for item in subbatch],
            }
            try:
                _log(f"broadcasting sub-batch of {len(subbatch)} requests, prompt_lens={[len(item.prompt_tokens) for item in subbatch]}")
                t0 = time.time()
                outputs, continuations = self._broadcast_and_generate(payload, stream_ctxs)
                elapsed = time.time() - t0
                total_gen = sum(len(o['completion_tokens']) for o in outputs)
                _log(f"sub-batch done in {elapsed:.1f}s, total_gen_tokens={total_gen}, tok/s={total_gen/max(elapsed,0.001):.1f}")
                # Build set of unfinished indices for continuation
                cont_indices = {c["index"] for c in continuations}
                for i, (item, output, sctx) in enumerate(zip(subbatch, outputs, stream_ctxs)):
                    if i in cont_indices:
                        continue  # handled below as continuation
                    if sctx is not None:
                        self._finalize_stream(item, output, sctx)
                    else:
                        item.pending.handle.set_result(self._build_response(item, output))
                # V2: merge unfinished items back as new sub-batches
                if continuations:
                    cont_prepared: List[PreparedRequest] = []
                    cont_stream_ctxs: List[Optional[StreamContext]] = []
                    for c in continuations:
                        idx = c["index"]
                        orig = subbatch[idx]
                        # Create continuation PreparedRequest with full token history
                        cont_req = PreparedRequest(
                            pending=orig.pending,
                            messages=orig.messages,
                            prompt_tokens=c["full_tokens"],
                            prompt_key=orig.prompt_key,
                            temperature=orig.temperature,
                            top_p=orig.top_p,
                            max_new_tokens=c["remaining_max_tokens"],
                            stop=orig.stop,
                        )
                        cont_prepared.append(cont_req)
                        # Carry forward stream context (update prompt_len to skip re-prefilled tokens)
                        orig_sctx = stream_ctxs[idx]
                        if orig_sctx is not None:
                            orig_sctx.prompt_len = len(c["full_tokens"])
                        cont_stream_ctxs.append(orig_sctx)
                    _log(f"V2 continuation: {len(cont_prepared)} unfinished items re-queued")
                    # These will be merged with new pending below
                    cont_subs = self._split_prepared_batches(cont_prepared)
                    # Prepend continuations so they run before new requests
                    # Attach their stream_ctxs via a side-channel
                    for csub in cont_subs:
                        pending_subbatches.insert(0, csub)
                    # Store continuation stream contexts for next iteration
                    if not hasattr(self, '_cont_stream_ctxs'):
                        self._cont_stream_ctxs = {}
                    for cp, cs in zip(cont_prepared, cont_stream_ctxs):
                        self._cont_stream_ctxs[id(cp)] = cs
            except Exception as exc:
                for item in subbatch:
                    item.pending.handle.set_error(exc)
            # V1 Continuous Batching: after each sub-batch, eagerly pull new pending requests
            new_pending: List[PendingRequest] = []
            with self.cv:
                now = int(time.time())
                while self.pending:
                    p = self.pending.pop(0)
                    if now - p.created > self.request_timeout_s:
                        p.handle.set_error(TimeoutError(f"request waited {now - p.created}s in queue, dropped"))
                    else:
                        new_pending.append(p)
            if new_pending:
                new_prepared: List[PreparedRequest] = []
                for p in new_pending:
                    try:
                        new_prepared.append(self._prepare_request(p))
                    except Exception as exc:
                        p.handle.set_error(exc)
                if new_prepared:
                    new_subs = self._split_prepared_batches(new_prepared)
                    _log(f"continuous batching: pulled {len(new_pending)} new requests → {len(new_subs)} sub-batches")
                    pending_subbatches.extend(new_subs)

    def _split_prepared_batches(self, prepared: List[PreparedRequest]) -> List[List[PreparedRequest]]:
        if len(prepared) <= 1:
            return [prepared]
        items = sorted(prepared, key=lambda item: (len(item.prompt_tokens), item.max_new_tokens))
        batches: List[List[PreparedRequest]] = []
        current: List[PreparedRequest] = []
        current_tokens = 0
        for item in items:
            item_tokens = len(item.prompt_tokens) + item.max_new_tokens
            over_batch = len(current) >= self.max_batch_size
            over_tokens = self.max_batch_total_tokens > 0 and current and (current_tokens + item_tokens > self.max_batch_total_tokens)
            if over_batch or over_tokens:
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(item)
            current_tokens += item_tokens
        if current:
            batches.append(current)
        return batches

    def _broadcast_and_generate(self, payload: Dict[str, Any],
                                 stream_ctxs: Optional[List[Optional[StreamContext]]] = None):
        payloads = [payload]
        dist.broadcast_object_list(payloads, src=0, group=self.ctrl_group)
        return self._distributed_generate(payload, stream_ctxs)

    def _sample_one(self, logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1)
        logits = logits / max(temperature, 1e-5)
        if top_p >= 1.0:
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
            return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1, dtype=torch.float32)
        cumulative = sorted_probs.cumsum(dim=-1)
        keep = cumulative <= top_p
        keep[..., 0] = True
        filtered = torch.where(keep, sorted_logits, torch.full_like(sorted_logits, float("-inf")))
        probs = torch.softmax(filtered, dim=-1, dtype=torch.float32)
        sampled = probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
        return sorted_indices.gather(1, sampled.unsqueeze(-1)).squeeze(-1)

    @torch.inference_mode()
    def _distributed_generate(self, payload: Dict[str, Any],
                               stream_ctxs: Optional[List[Optional[StreamContext]]] = None) -> List[Dict[str, Any]]:
        self.model.reset_caches(release=False)
        _gen_start = time.perf_counter()
        prompt_tokens: List[List[int]] = payload["prompt_tokens"]
        temperatures: List[float] = payload["temperatures"]
        top_ps: List[float] = payload["top_ps"]
        max_new_tokens: List[int] = payload["max_new_tokens"]
        try:
            bsz = len(prompt_tokens)
            prompt_lens = [len(t) for t in prompt_tokens]
            total_len = min(self.model.max_seq_len, max(p + m for p, m in zip(prompt_lens, max_new_tokens)))
            if self.global_rank == 0:
                _log(f"generate bsz={bsz}, prompt_lens={prompt_lens[:5]}{'...' if bsz>5 else ''}, total_len={total_len}, max_seq_len={self.model.max_seq_len}")
            tokens = torch.full((bsz, total_len), -1, dtype=torch.long, device="cuda")
            for i, prompt in enumerate(prompt_tokens):
                tokens[i, :len(prompt)] = torch.tensor(prompt, dtype=torch.long, device="cuda")
            prompt_mask = tokens != -1
            finished = torch.zeros(bsz, dtype=torch.bool, device="cuda")
            generation_ends = torch.tensor([min(total_len, p + m) for p, m in zip(prompt_lens, max_new_tokens)], dtype=torch.long, device="cuda")
            # Pre-allocate reusable tensors for the decode loop
            forced_eos = torch.full((bsz,), self.tokenizer.eos_token_id, dtype=torch.long, device="cuda")
            eos_id = self.tokenizer.eos_token_id
            max_prompt_len = max(prompt_lens)
            # Pre-allocate PP communication buffers for decode (seqlen=1)
            _pp_h_buf = torch.empty(bsz, 1, self.hc_mult, self.dim, dtype=torch.bfloat16, device="cuda") if self.pp_rank != 0 else None
            _pp_tok_buf = torch.empty(bsz, dtype=torch.long, device="cuda") if self.pp_rank == 0 else None
            prev_pos = 0
            shared_prefill = min(prompt_lens)
            logits = None
            while prev_pos + self.prefill_chunk_size < shared_prefill:
                cur_pos = prev_pos + self.prefill_chunk_size
                input_ids = tokens[:, prev_pos:cur_pos]
                logits = pp_forward(
                    self.model,
                    input_ids,
                    prev_pos,
                    self.pp_rank,
                    self.pp_peer_rank,
                    self.hc_mult,
                    self.dim,
                    self.vocab_size,
                )
                prev_pos = cur_pos
            # Decode loop — optimized for the common bsz==1 pure-decode case
            _pure_decode = (bsz == 1 and shared_prefill == max_prompt_len)
            _gen_end_val = generation_ends[0].item() if _pure_decode else 0
            _step_times = []
            for cur_pos in range(shared_prefill, total_len):
                _t0 = time.perf_counter()
                input_ids = tokens[:, prev_pos:cur_pos]
                next_token = pp_next_token(
                    self.model,
                    input_ids,
                    prev_pos,
                    self.pp_rank,
                    self.pp_peer_rank,
                    self.hc_mult,
                    self.dim,
                    self.vocab_size,
                    temperatures,
                    top_ps,
                    33377335 + cur_pos,
                    h_buf=_pp_h_buf,
                    tok_buf=_pp_tok_buf,
                )
                _step_times.append(time.perf_counter() - _t0)
                if _pure_decode:
                    # Fast path: bsz=1, all positions are decode — skip prompt_mask checks
                    tokens[0, cur_pos] = next_token[0]
                    tok_val = next_token[0].item()
                    if tok_val == eos_id or cur_pos + 1 >= _gen_end_val:
                        finished[0] = True
                        # Push final streaming token before break
                        if stream_ctxs is not None:
                            sctx = stream_ctxs[0]
                            if sctx is not None and not sctx.finished:
                                if tok_val == eos_id:
                                    sctx.finished = True
                                else:
                                    sctx.gen_tokens.append(tok_val)
                                    new_text = self.tokenizer.decode(sctx.gen_tokens, skip_special_tokens=True)
                                    delta = new_text[len(sctx.prev_text):]
                                    if delta:
                                        sctx.handle.push_token(delta)
                                        sctx.prev_text = new_text
                        prev_pos = cur_pos
                        break
                    if stream_ctxs is not None:
                        sctx = stream_ctxs[0]
                        if sctx is not None and not sctx.finished:
                            sctx.gen_tokens.append(tok_val)
                            new_text = self.tokenizer.decode(sctx.gen_tokens, skip_special_tokens=True)
                            delta = new_text[len(sctx.prev_text):]
                            if delta:
                                sctx.handle.push_token(delta)
                                sctx.prev_text = new_text
                else:
                    # General batched path
                    next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
                    active = cur_pos < generation_ends
                    next_token = torch.where(prompt_mask[:, cur_pos] | active, next_token, forced_eos)
                    tokens[:, cur_pos] = next_token
                    finished |= ((~prompt_mask[:, cur_pos]) & (next_token == eos_id)) | (cur_pos + 1 >= generation_ends)
                    if stream_ctxs is not None:
                        next_token_list = next_token.tolist()
                        for i, sctx in enumerate(stream_ctxs):
                            if sctx is None or sctx.finished:
                                continue
                            if cur_pos < prompt_lens[i]:
                                continue
                            tok_id = next_token_list[i]
                            if tok_id == eos_id:
                                sctx.finished = True
                                continue
                            sctx.gen_tokens.append(tok_id)
                            new_text = self.tokenizer.decode(sctx.gen_tokens, skip_special_tokens=True)
                            delta = new_text[len(sctx.prev_text):]
                            if delta:
                                sctx.handle.push_token(delta)
                                sctx.prev_text = new_text
                    if finished.all():
                        break
                    # V2: early break when enough items finish + pending requests exist
                    # Only break when re-prefill cost is justified:
                    #   - at least half the batch is done (avoids breaking for 1/19)
                    #   - re-prefill cost (n_unfinished * cur_pos) < 2x remaining decode work
                    if bsz > 1:
                        decode_step = cur_pos - shared_prefill + 1
                        if decode_step > 0 and decode_step % 64 == 0:
                            n_finished_v2 = finished.sum().item()
                            if n_finished_v2 >= max(bsz // 2, 2):
                                n_unfinished = bsz - n_finished_v2
                                reprefill_cost = n_unfinished * cur_pos
                                remaining_decode = n_unfinished * max(total_len - cur_pos, 1)
                                if reprefill_cost <= remaining_decode * 2:
                                    should_break = [False]
                                    if self.global_rank == 0:
                                        with self.cv:
                                            should_break[0] = len(self.pending) > 0
                                    dist.broadcast_object_list(should_break, src=0, group=self.ctrl_group)
                                    if should_break[0]:
                                        if self.global_rank == 0:
                                            _log(f"V2 early break at pos {cur_pos}: {n_finished_v2}/{bsz} done, "
                                                 f"reprefill={reprefill_cost} vs remaining={remaining_decode}")
                                        break
                prev_pos = cur_pos
            # Log step timing diagnostics
            if _step_times and self.global_rank == 0:
                n = len(_step_times)
                first = _step_times[0] * 1000
                if n > 1:
                    decode_times = _step_times[1:]
                    avg = sum(decode_times) / len(decode_times) * 1000
                    _log(f"step timing: first={first:.0f}ms (prefill+decode), decode avg={avg:.1f}ms/tok ({1000/avg:.1f} tok/s), steps={n}")
                else:
                    _log(f"step timing: first={first:.0f}ms, steps={n}")
            # Build outputs + continuation info for unfinished items
            early_break = not finished.all()
            outputs: List[Dict[str, Any]] = []
            continuations: List[Dict[str, Any]] = []
            completion_tokens_list = tokens.tolist()
            for i, full_tokens in enumerate(completion_tokens_list):
                start = prompt_lens[i]
                end = min(len(full_tokens), prompt_lens[i] + max_new_tokens[i])
                piece = full_tokens[start:end]
                if self.tokenizer.eos_token_id in piece:
                    piece = piece[:piece.index(self.tokenizer.eos_token_id)]
                piece.append(self.tokenizer.eos_token_id)
                gen_time = time.perf_counter() - _gen_start
                outputs.append({"completion_tokens": piece, "prompt_tokens": prompt_lens[i], "generation_time_s": gen_time})
                if early_break and not finished[i]:
                    # Collect full token history for continuation
                    history = [t for t in full_tokens if t != -1]
                    remaining = max_new_tokens[i] - (len(history) - prompt_lens[i])
                    continuations.append({
                        "index": i,
                        "full_tokens": history,
                        "remaining_max_tokens": max(1, remaining),
                    })
            return outputs, continuations
        finally:
            self.model.reset_caches(release=self.release_kv_after_batch)

    def _apply_stop(self, text: str, stop: Optional[Union[str, List[str]]]) -> str:
        if stop is None:
            return text
        stops = [stop] if isinstance(stop, str) else stop
        cut = None
        for item in stops:
            idx = text.find(item)
            if idx >= 0:
                cut = idx if cut is None else min(cut, idx)
        if cut is None:
            return text
        return text[:cut]

    def _finalize_stream(self, prepared: PreparedRequest, output: Dict[str, Any], sctx: StreamContext):
        """Send the finish signal to a streaming handle after generation completes."""
        completion_token_count = len(sctx.gen_tokens)
        gen_time = output.get("generation_time_s", 0)
        gen_tps = completion_token_count / max(gen_time, 0.001) if completion_token_count > 0 else 0
        usage = {
            "prompt_tokens": output["prompt_tokens"],
            "completion_tokens": completion_token_count,
            "total_tokens": output["prompt_tokens"] + completion_token_count,
            "generation_time_s": round(gen_time, 3),
            "tokens_per_second": round(gen_tps, 1),
        }
        sctx.handle.push_finish("stop", usage)

    def _build_response(self, prepared: PreparedRequest, output: Dict[str, Any]) -> Dict[str, Any]:
        raw_text = self.tokenizer.decode(output["completion_tokens"], skip_special_tokens=False)
        content = raw_text
        parsed = None
        try:
            parsed = parse_message_from_completion_text(raw_text, thinking_mode="chat")
            content = parsed.get("content") or ""
        except Exception:
            content = raw_text.replace(self.tokenizer.eos_token or "", "")
        content = self._apply_stop(content, prepared.stop)
        finish_reason = "stop"
        message: Dict[str, Any] = {"role": "assistant", "content": content}
        if parsed and parsed.get("tool_calls"):
            message["tool_calls"] = parsed["tool_calls"]
            finish_reason = "tool_calls"
        completion_token_count = len(self.tokenizer.encode(content, add_special_tokens=False))
        gen_time = output.get("generation_time_s", 0)
        gen_tps = completion_token_count / max(gen_time, 0.001) if completion_token_count > 0 else 0
        return {
            "id": prepared.pending.request_id,
            "object": "chat.completion",
            "created": prepared.pending.created,
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": output["prompt_tokens"],
                "completion_tokens": completion_token_count,
                "total_tokens": output["prompt_tokens"] + completion_token_count,
                "generation_time_s": round(gen_time, 3),
                "tokens_per_second": round(gen_tps, 1),
            },
        }


def build_true_stream_response(handle: RequestHandle, request_id: str, created: int, model_name: str):
    """True token-by-token SSE streaming that reads from the handle's token_queue."""
    async def gen():
        # First chunk: role
        head = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(head, ensure_ascii=False)}\n\n"
        # Read tokens from queue until finish or error
        while True:
            try:
                msg = await asyncio.to_thread(handle.token_queue.get, timeout=120)
            except Exception:
                break
            msg_type = msg[0]
            if msg_type == "token":
                text = msg[1]
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            elif msg_type == "finish":
                info = msg[1]
                tail = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": info["finish_reason"]}],
                    "usage": info.get("usage"),
                }
                yield f"data: {json.dumps(tail, ensure_ascii=False)}\n\n"
                break
            elif msg_type == "error":
                break
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


def create_app(engine: BatchedOpenAIServer) -> FastAPI:
    app = FastAPI()

    @app.on_event("startup")
    async def _setup_thread_pool():
        loop = asyncio.get_event_loop()
        loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=256)
        )
        _log("thread pool executor set to 256 workers")

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": engine.model_name}

    @app.get("/v1/models")
    async def models():
        created = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.model_name,
                    "object": "model",
                    "created": created,
                    "owned_by": "deepseekV4-flash",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(body: ChatCompletionRequest):
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        handle = engine.submit(body, request_id=request_id, created=created)
        if body.stream:
            return build_true_stream_response(handle, request_id, created, engine.model_name)
        handle.event.wait()
        if handle.error is not None:
            raise HTTPException(status_code=400, detail=str(handle.error))
        return JSONResponse(handle.result)

    return app


def _warmup_pp(pp_rank, pp_peer_rank):
    """Pre-create NCCL P2P communicators so the first real request doesn't pay the lazy-init cost.
    Only warms up the P2P channels — avoids running model forward (which triggers slow JIT kernel compilation).
    Called by ALL ranks from init_runtime (before the worker/scheduler split)."""
    _log("warming up PP communicators...")
    dummy = torch.zeros(1, dtype=torch.bfloat16, device="cuda")
    # Round 1: stage0→stage1
    if pp_rank == 0:
        dist.send(dummy.contiguous(), dst=pp_peer_rank)
        dist.recv(dummy, src=pp_peer_rank)
    else:
        dist.recv(dummy, src=pp_peer_rank)
        dist.send(dummy.contiguous(), dst=pp_peer_rank)
    # Round 2: stage1→stage0 (reverse direction, creates the other communicator)
    if pp_rank == 0:
        dist.recv(dummy, src=pp_peer_rank)
        dist.send(dummy.contiguous(), dst=pp_peer_rank)
    else:
        dist.send(dummy.contiguous(), dst=pp_peer_rank)
        dist.recv(dummy, src=pp_peer_rank)
    torch.cuda.synchronize()
    _log("PP warmup done")


def init_runtime(ckpt_path: str, config_path: str, max_batch_size: int, max_seq_len_override: int):
    global_world_size = int(os.getenv("WORLD_SIZE", "1"))
    global_rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if global_world_size != 16:
        raise RuntimeError(f"PP=2 x TP=8 requires exactly 16 GPUs, got {global_world_size}")
    dist.init_process_group("nccl", timeout=timedelta(hours=12))
    pp_size = 2
    pp_rank = global_rank // TP_SIZE
    tp_rank = global_rank % TP_SIZE
    if pp_rank == 0:
        pp_peer_rank = global_rank + TP_SIZE
    else:
        pp_peer_rank = global_rank - TP_SIZE
    stage0_ranks = list(range(0, TP_SIZE))
    stage1_ranks = list(range(TP_SIZE, 2 * TP_SIZE))
    tp_group_0 = dist.new_group(stage0_ranks)
    tp_group_1 = dist.new_group(stage1_ranks)
    tp_grp = tp_group_0 if pp_rank == 0 else tp_group_1
    ctrl_group = dist.new_group(backend="gloo", timeout=timedelta(hours=12))
    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)
    with open(config_path, "r", encoding="utf-8") as f:
        args = ModelArgs(**json.load(f))
    args.max_batch_size = max_batch_size
    if max_seq_len_override > 0:
        args.max_seq_len = max_seq_len_override
    with torch.device("cuda"):
        model = Transformer(args, pp_rank=pp_rank, pp_size=pp_size, tp_world_size=TP_SIZE, tp_rank=tp_rank, tp_grp=tp_grp)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    shard_file = os.path.join(ckpt_path, f"model{tp_rank}-mp{TP_SIZE}.safetensors")
    load_pp_weights(model, shard_file, pp_rank, pp_size)
    torch.set_default_device("cuda")
    _warmup_pp(pp_rank, pp_peer_rank)
    return model, tokenizer, args, ctrl_group, global_rank, pp_rank, pp_peer_rank


def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", type=str, default="deepseek-v4-flash")
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--batch-timeout-ms", type=int, default=5000)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--max-batch-total-tokens", type=int, default=0)
    parser.add_argument("--release-kv-after-batch", action="store_true")
    parser.add_argument("--max-queue-size", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=0)
    parser.add_argument("--request-timeout-s", type=int, default=600, help="Drop queued requests older than this (seconds)")
    args = parser.parse_args()

    model, tokenizer, model_args, ctrl_group, global_rank, pp_rank, pp_peer_rank = init_runtime(
        args.ckpt_path,
        args.config,
        args.max_batch_size,
        args.max_seq_len,
    )
    engine = BatchedOpenAIServer(
        model=model,
        tokenizer=tokenizer,
        ctrl_group=ctrl_group,
        global_rank=global_rank,
        pp_rank=pp_rank,
        pp_peer_rank=pp_peer_rank,
        hc_mult=model_args.hc_mult,
        dim=model_args.dim,
        vocab_size=model_args.vocab_size,
        max_batch_size=model_args.max_batch_size,
        batch_timeout_ms=args.batch_timeout_ms,
        prefill_chunk_size=args.prefill_chunk_size,
        max_batch_total_tokens=args.max_batch_total_tokens,
        release_kv_after_batch=args.release_kv_after_batch,
        model_name=args.model_name,
        max_queue_size=args.max_queue_size,
        request_timeout_s=args.request_timeout_s,
    )
    if global_rank == 0:
        engine.start()
        app = create_app(engine)
        try:
            uvicorn.run(
                app, host=args.host, port=args.port, workers=1,
                loop="uvloop" if _has_uvloop() else "auto",
                http="httptools" if _has_httptools() else "auto",
            )
        finally:
            engine.shutdown()
            payloads = [{"type": "shutdown"}]
            dist.broadcast_object_list(payloads, src=0, group=ctrl_group)
            dist.destroy_process_group()
    else:
        try:
            engine.worker_loop()
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
