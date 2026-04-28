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
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import safe_open

from model import Transformer, ModelArgs

current_dir = os.path.dirname(os.path.abspath(__file__))
encoding_dir = os.path.join(current_dir, '../encoding')
sys.path.insert(0, os.path.abspath(encoding_dir))
from encoding_dsv4 import encode_messages, parse_message_from_completion_text


TP_SIZE = 8  # tensor parallelism (within each node)


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
        logits = pp_forward(model, input_ids, prev_pos, pp_rank, pp_peer_rank, hc_mult, dim, vocab_size)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
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
                                        dim, vocab_size, temperature)
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
                                    dim, vocab_size, temperature)
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
    args = parser.parse_args()
    assert args.input_file or args.interactive
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature)
