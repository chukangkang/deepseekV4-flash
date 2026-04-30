"""
Pipeline-Parallel (PP=2) x Tensor-Parallel (TP=8) inference for DeepSeek-V4.

Launch with WORLD_SIZE=16 (2 nodes x 8 GPUs).
  - Global ranks 0-7  → Node 0 → pipeline stage 0 (embed + first half of layers)
  - Global ranks 8-15 → Node 1 → pipeline stage 1 (second half of layers + head)

Each pipeline stage runs TP=8 within its node.
Inter-stage communication uses point-to-point send/recv between matching TP ranks.
"""
import os
import json
import sys
from argparse import ArgumentParser
from datetime import timedelta
from typing import List, Sequence

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import safe_open

from model import Transformer, ModelArgs

current_dir = os.path.dirname(os.path.abspath(__file__))
encoding_dir = os.path.join(current_dir, '../encoding')
sys.path.insert(0, os.path.abspath(encoding_dir))
from encoding_dsv4 import encode_messages, parse_message_from_completion_text


TP_SIZE = 8


def _remap_key(ckpt_key: str, layer_start: int, layer_end: int, pp_rank: int, pp_size: int):
    """Map a checkpoint key to a model state_dict key, or return None to skip."""
    if ckpt_key.startswith("layers."):
        parts = ckpt_key.split(".", 2)  # ['layers', '{id}', 'rest...']
        orig_layer_id = int(parts[1])
        if orig_layer_id < layer_start or orig_layer_id >= layer_end:
            return None
        local_id = orig_layer_id - layer_start
        return f"layers.{local_id}.{parts[2]}"
    elif ckpt_key.startswith("embed."):
        if pp_rank == 0:
            return ckpt_key
        elif pp_size > 1 and pp_rank == pp_size - 1:
            return ckpt_key.replace("embed.", "mtp_embed.", 1)
        return None
    elif ckpt_key.startswith("head.") or ckpt_key.startswith("norm.") or \
         ckpt_key.startswith("mtp.") or ckpt_key.startswith("hc_head"):
        if pp_rank != pp_size - 1:
            return None
        return ckpt_key
    return ckpt_key


def load_pp_weights(model: Transformer, shard_file: str, pp_rank: int, pp_size: int):
    """Load weights from a safetensors shard with PP layer index remapping.
    
    Only loads tensors that belong to this pipeline stage (filters by key
    BEFORE calling get_tensor to avoid unnecessary GPU memory allocation).
    
    Remaps checkpoint 'layers.{orig_id}.*' → model 'layers.{local_id}.*'.
    """
    layer_start = model.layer_start
    layer_end = model.layer_end
    
    remapped = {}
    skipped = 0
    with safe_open(shard_file, framework="pt", device="cpu") as f:
        for ckpt_key in f.keys():
            model_key = _remap_key(ckpt_key, layer_start, layer_end, pp_rank, pp_size)
            if model_key is None:
                skipped += 1
                continue
            # Only load tensors we actually need
            remapped[model_key] = f.get_tensor(ckpt_key)
    
    result = model.load_state_dict(remapped, strict=False)
    loaded = len(remapped) - len(result.unexpected_keys)
    return loaded, skipped + len(result.unexpected_keys)


def sample(logits, temperature: float = 1.0):
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


def sample_with_top_p(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0, seed: int = 0):
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    generator = torch.Generator(device=logits.device)
    generator.manual_seed(seed)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        keep = cumulative <= top_p
        keep[..., 0] = True
        sorted_probs = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min_(1e-12)
        noise = torch.rand(sorted_probs.shape, generator=generator, device=sorted_probs.device).clamp_(1e-12, 1.0)
        sampled = sorted_probs.div_(-noise.log_()).argmax(dim=-1)
        return sorted_indices.gather(1, sampled.unsqueeze(-1)).squeeze(-1)
    noise = torch.rand(probs.shape, generator=generator, device=probs.device).clamp_(1e-12, 1.0)
    return probs.div_(-noise.log_()).argmax(dim=-1)


def sample_batch(logits: torch.Tensor, temperatures: Sequence[float], top_ps: Sequence[float], seed: int):
    next_token = torch.empty(logits.size(0), dtype=torch.long, device=logits.device)
    for row in range(logits.size(0)):
        next_token[row] = sample_with_top_p(logits[row:row + 1], temperatures[row], top_ps[row], seed + row)
    return next_token


def pp_forward(model, input_ids, start_pos, pp_rank, pp_peer_rank, hc_mult, dim, vocab_size):
    """One forward step with pipeline parallelism.
    Stage 0: embed + layers → send hidden [bsz, seqlen, hc_mult, dim] (bf16) to stage 1
    Stage 1: recv hidden → layers + head → send logits [bsz, vocab_size] (float32) to stage 0
    Returns logits (float32) on all ranks."""
    bsz, seqlen = input_ids.size()
    if pp_rank == 0:
        h = model.forward(input_ids, start_pos)  # [bsz, seqlen, hc_mult, dim] bf16
        dist.send(h.contiguous(), dst=pp_peer_rank)
        logits = torch.empty(bsz, vocab_size, dtype=torch.float32, device="cuda")
        dist.recv(logits, src=pp_peer_rank)
        return logits
    else:
        h = torch.empty(bsz, seqlen, hc_mult, dim, dtype=torch.bfloat16, device="cuda")
        dist.recv(h, src=pp_peer_rank)
        logits = model.forward(input_ids, start_pos, hidden_states=h)  # [bsz, vocab_size] float32
        dist.send(logits.contiguous(), dst=pp_peer_rank)
        return logits


_pp_prof_steps = 0
_pp_prof_fwd_total = 0.0
_pp_prof_wait_total = 0.0
_pp_prof_fwd_window = 0.0
_pp_prof_wait_window = 0.0
_PP_PROF_INTERVAL = 50
_PP_PROF_ENABLED = os.environ.get("PP_PROFILE", "0") == "1"
_PP_PIPELINE = os.environ.get("PP_PIPELINE", "1") == "1"
_PP_PIPELINE_MIN_BSZ = int(os.environ.get("PP_PIPELINE_MIN_BSZ", "6"))


# ---------------------------------------------------------------------------
# KV cache batch-offset helpers for micro-batch pipeline
# ---------------------------------------------------------------------------

def _offset_cache_mgr(mgr, offset):
    """Shift DynamicKVCache or TurboQuantKVCache storage along batch dim."""
    # DynamicKVCache path
    if hasattr(mgr, 'flat_storage') and mgr.flat_storage is not None:
        if offset > 0:
            if not hasattr(mgr, '_pp_orig_flat'):
                mgr._pp_orig_flat = mgr.flat_storage
            mgr.flat_storage = mgr._pp_orig_flat[offset:]
        elif hasattr(mgr, '_pp_orig_flat'):
            mgr.flat_storage = mgr._pp_orig_flat
            del mgr._pp_orig_flat
    # TurboQuantKVCache path
    for attr in ('quantized', 'scales', 'zeros', '_bf16_view'):
        val = getattr(mgr, attr, None)
        if val is None:
            continue
        orig_key = f'_pp_orig_{attr}'
        if offset > 0:
            if not hasattr(mgr, orig_key):
                setattr(mgr, orig_key, val)
            setattr(mgr, attr, getattr(mgr, orig_key)[offset:])
        elif hasattr(mgr, orig_key):
            setattr(mgr, attr, getattr(mgr, orig_key))
            delattr(mgr, orig_key)


def _offset_compressor(comp, offset):
    """Shift Compressor kv_state / score_state along batch dim."""
    if offset > 0:
        if not hasattr(comp, '_pp_orig_kv_state'):
            comp._pp_orig_kv_state = comp.kv_state
            comp._pp_orig_score_state = comp.score_state
        comp.kv_state = comp._pp_orig_kv_state[offset:]
        comp.score_state = comp._pp_orig_score_state[offset:]
    elif hasattr(comp, '_pp_orig_kv_state'):
        comp.kv_state = comp._pp_orig_kv_state
        comp.score_state = comp._pp_orig_score_state
        del comp._pp_orig_kv_state, comp._pp_orig_score_state


def _set_kv_batch_offset(model, offset):
    """Shift every layer's KV cache / compressor / indexer views so that
    batch index 0 in the model maps to physical batch slot `offset`.
    Call with offset=0 to restore original views."""
    for layer in model.layers:
        attn = layer.attn
        _offset_cache_mgr(attn.kv_cache_mgr, offset)
        if hasattr(attn, 'compressor') and attn.compressor is not None:
            _offset_compressor(attn.compressor, offset)
        if hasattr(attn, 'indexer') and attn.indexer is not None:
            _offset_cache_mgr(attn.indexer.kv_cache_mgr, offset)
            if hasattr(attn.indexer, 'compressor'):
                _offset_compressor(attn.indexer.compressor, offset)


def _pp_next_token_pipelined(model, input_ids, start_pos, pp_rank, pp_peer_rank,
                              hc_mult, dim, vocab_size, temperatures, top_ps, seed):
    """PP decode with 2 micro-batches: overlap stage0-fwd(B) with stage1-fwd(A).

    Timeline (F=fwd_half, C=comm):
      Stage0: [fwd_A][isend hA + fwd_B][send hB]...[recv tokA][recv tokB]
      Stage1: .......[recv hA][fwd_A + irecv hB][send tokA][fwd_B][send tokB]
                              └─── overlap ────┘
      Total ≈ 3·F_half + 2·C   vs  current  2·F_full + 2·C
    """
    bsz = input_ids.size(0)
    mid = bsz // 2
    ids_A, ids_B = input_ids[:mid], input_ids[mid:]
    temps_A, temps_B = temperatures[:mid], temperatures[mid:]
    top_ps_A, top_ps_B = top_ps[:mid], top_ps[mid:]
    if isinstance(start_pos, torch.Tensor):
        sp_A, sp_B = start_pos[:mid], start_pos[mid:]
    else:
        sp_A = sp_B = start_pos

    if pp_rank == 0:
        # --- micro-batch A (batch slots 0..mid-1) ---
        h_A = model.forward(ids_A, sp_A)
        # Async send h_A so we can start fwd_B immediately
        req_send_A = dist.isend(h_A.contiguous(), dst=pp_peer_rank)
        # --- micro-batch B (batch slots mid..bsz-1) ---
        _set_kv_batch_offset(model, mid)
        h_B = model.forward(ids_B, sp_B)
        _set_kv_batch_offset(model, 0)
        # Finish send A, then send B
        req_send_A.wait()
        dist.send(h_B.contiguous(), dst=pp_peer_rank)
        # Receive tokens (two separate recvs to match stage1's two sends)
        next_token = torch.empty(bsz, dtype=torch.long, device="cuda")
        dist.recv(next_token[:mid], src=pp_peer_rank)
        dist.recv(next_token[mid:], src=pp_peer_rank)
        return next_token
    else:
        # --- recv h_A (blocking), start async recv h_B ---
        h_A = torch.empty(mid, 1, hc_mult, dim, dtype=torch.bfloat16, device="cuda")
        dist.recv(h_A, src=pp_peer_rank)
        h_B = torch.empty(bsz - mid, 1, hc_mult, dim, dtype=torch.bfloat16, device="cuda")
        req_recv_B = dist.irecv(h_B, src=pp_peer_rank)
        # --- fwd A (overlaps with h_B arriving on NCCL stream) ---
        logits_A = model.forward(ids_A, sp_A, hidden_states=h_A)
        tok_A = sample_batch(logits_A, temps_A, top_ps_A, seed)
        dist.send(tok_A.contiguous(), dst=pp_peer_rank)
        # --- fwd B ---
        req_recv_B.wait()
        _set_kv_batch_offset(model, mid)
        logits_B = model.forward(ids_B, sp_B, hidden_states=h_B)
        _set_kv_batch_offset(model, 0)
        tok_B = sample_batch(logits_B, temps_B, top_ps_B, seed + mid)
        dist.send(tok_B.contiguous(), dst=pp_peer_rank)
        return torch.cat([tok_A, tok_B])


def pp_next_token(model, input_ids, start_pos, pp_rank, pp_peer_rank, hc_mult, dim, vocab_size,
                  temperatures, top_ps, seed: int, h_buf=None, tok_buf=None):
    global _pp_prof_steps, _pp_prof_fwd_total, _pp_prof_wait_total, _pp_prof_fwd_window, _pp_prof_wait_window
    bsz, seqlen = input_ids.size()
    # Micro-batch pipeline for decode with enough batch items
    if _PP_PIPELINE and seqlen == 1 and bsz >= _PP_PIPELINE_MIN_BSZ:
        return _pp_next_token_pipelined(
            model, input_ids, start_pos, pp_rank, pp_peer_rank,
            hc_mult, dim, vocab_size, temperatures, top_ps, seed)
    _do_profile = _PP_PROF_ENABLED and seqlen == 1  # only profile decode steps
    if pp_rank == 0:
        if _do_profile:
            torch.cuda.synchronize()
            import time; _t0 = time.perf_counter()
        h = model.forward(input_ids, start_pos)
        if _do_profile:
            torch.cuda.synchronize()
            _t1 = time.perf_counter()
        dist.send(h.contiguous(), dst=pp_peer_rank)
        if tok_buf is not None and tok_buf.shape[0] >= bsz:
            next_token = tok_buf[:bsz]
        else:
            next_token = torch.empty(bsz, dtype=torch.long, device="cuda")
        dist.recv(next_token, src=pp_peer_rank)
        if _do_profile:
            torch.cuda.synchronize()  # wait for NCCL recv to actually complete
            _t2 = time.perf_counter()
            fwd_ms = (_t1 - _t0) * 1000
            wait_ms = (_t2 - _t1) * 1000
            _pp_prof_fwd_total += fwd_ms
            _pp_prof_wait_total += wait_ms
            _pp_prof_fwd_window += fwd_ms
            _pp_prof_wait_window += wait_ms
            _pp_prof_steps += 1
            if _pp_prof_steps % _PP_PROF_INTERVAL == 0:
                avg_fwd = _pp_prof_fwd_total / _pp_prof_steps
                avg_wait = _pp_prof_wait_total / _pp_prof_steps
                win_fwd = _pp_prof_fwd_window / _PP_PROF_INTERVAL
                win_wait = _pp_prof_wait_window / _PP_PROF_INTERVAL
                print(f"[pp_profile] stage0 step {_pp_prof_steps}: "
                      f"last{_PP_PROF_INTERVAL} fwd={win_fwd:.0f}ms wait={win_wait:.0f}ms total={win_fwd+win_wait:.0f}ms | "
                      f"cumul fwd={avg_fwd:.0f}ms wait={avg_wait:.0f}ms total={avg_fwd+avg_wait:.0f}ms",
                      flush=True)
                _pp_prof_fwd_window = 0.0
                _pp_prof_wait_window = 0.0
        return next_token
    if h_buf is not None and h_buf.shape[0] >= bsz and h_buf.shape[1] >= seqlen:
        h = h_buf[:bsz, :seqlen]
    else:
        h = torch.empty(bsz, seqlen, hc_mult, dim, dtype=torch.bfloat16, device="cuda")
    dist.recv(h, src=pp_peer_rank)
    if _do_profile:
        torch.cuda.synchronize()
        import time; _t0 = time.perf_counter()
    logits = model.forward(input_ids, start_pos, hidden_states=h)
    next_token = sample_batch(logits, temperatures, top_ps, seed)
    if _do_profile:
        torch.cuda.synchronize()
        _t1 = time.perf_counter()
    dist.send(next_token.contiguous(), dst=pp_peer_rank)
    return next_token


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    pp_rank: int,
    pp_peer_rank: int,
    hc_mult: int,
    dim: int,
    vocab_size: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> List[List[int]]:
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    prev_pos = 0
    finished = torch.zeros(len(prompt_tokens), dtype=torch.bool, device="cuda")
    prompt_mask = tokens != -1
    for cur_pos in range(min(prompt_lens), total_len):
        input_ids = tokens[:, prev_pos:cur_pos]
        next_token = pp_next_token(
            model,
            input_ids,
            prev_pos,
            pp_rank,
            pp_peer_rank,
            hc_mult,
            dim,
            vocab_size,
            [temperature] * len(prompt_tokens),
            [top_p] * len(prompt_tokens),
            33377335 + cur_pos,
        )
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        toks.append(eos_id)
        completion_tokens.append(toks)
    return completion_tokens


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> None:
    # Global distributed setup
    global_world_size = int(os.getenv("WORLD_SIZE", "1"))
    global_rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    assert global_world_size == 16, f"PP=2 x TP=8 requires exactly 16 GPUs, got {global_world_size}"

    dist.init_process_group("nccl", timeout=timedelta(hours=12))

    # Pipeline parallelism config
    pp_size = 2
    pp_rank = global_rank // TP_SIZE  # 0 for ranks 0-7, 1 for ranks 8-15
    tp_rank = global_rank % TP_SIZE   # 0-7 within each node

    # PP peer: matching TP rank on the other stage
    if pp_rank == 0:
        pp_peer_rank = global_rank + TP_SIZE  # rank i ↔ rank i+8
    else:
        pp_peer_rank = global_rank - TP_SIZE  # rank i ↔ rank i-8

    # Create TP sub-group (ranks within same node)
    stage0_ranks = list(range(0, TP_SIZE))
    stage1_ranks = list(range(TP_SIZE, 2 * TP_SIZE))
    tp_group_0 = dist.new_group(stage0_ranks)
    tp_group_1 = dist.new_group(stage1_ranks)
    tp_grp = tp_group_0 if pp_rank == 0 else tp_group_1
    ctrl_group = dist.new_group(backend="gloo", timeout=timedelta(hours=12))

    global print
    if global_rank != 0:
        print = lambda *_, **__: None

    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)

    with open(config) as f:
        args = ModelArgs(**json.load(f))
    if interactive:
        args.max_batch_size = 1
    print(args)

    print(f"[Rank {global_rank}] PP stage {pp_rank}, TP rank {tp_rank}, peer {pp_peer_rank}")

    with torch.device("cuda"):
        model = Transformer(args, pp_rank=pp_rank, pp_size=pp_size,
                           tp_world_size=TP_SIZE, tp_rank=tp_rank, tp_grp=tp_grp)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    # Load weights: each TP rank loads its own shard (model{tp_rank}-mp{TP_SIZE}.safetensors)
    # Custom loader remaps layer indices for PP and routes embed/head to correct stage.
    shard_file = os.path.join(ckpt_path, f"model{tp_rank}-mp{TP_SIZE}.safetensors")
    print(f"[Rank {global_rank}] Loading {shard_file} (PP stage {pp_rank}, layers {model.layer_start}-{model.layer_end-1})")
    loaded, skipped = load_pp_weights(model, shard_file, pp_rank, pp_size)
    print(f"[Rank {global_rank}] Loaded {loaded} tensors, skipped {skipped}")

    torch.set_default_device("cuda")
    print("I'm DeepSeek 👋 (PP=2 x TP=8 mode)")

    hc_mult = args.hc_mult
    dim = args.dim
    vocab_size = args.vocab_size

    if interactive:
        messages = []
        while True:
            if global_rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0, group=ctrl_group)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0, group=ctrl_group)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.encode(encode_messages(messages, thinking_mode="chat"))
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens,
                                        tokenizer.eos_token_id,
                                        pp_rank, pp_peer_rank, hc_mult,
                                        dim, vocab_size, temperature, top_p)
            completion = tokenizer.decode(completion_tokens[0])
            print(completion)
            messages.append(parse_message_from_completion_text(completion, thinking_mode="chat"))
    else:
        with open(input_file) as f:
            prompts = f.read().split("\n\n")
        prompt_tokens = [tokenizer.encode(encode_messages([{"role": "user", "content": prompt}], thinking_mode="chat")) for prompt in prompts]
        completion_tokens = generate(model, prompt_tokens, max_new_tokens,
                                    tokenizer.eos_token_id,
                                    pp_rank, pp_peer_rank, hc_mult,
                                    dim, vocab_size, temperature, top_p)
        completions = tokenizer.batch_decode(completion_tokens)
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()
    assert args.input_file or args.interactive
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature, args.top_p)
