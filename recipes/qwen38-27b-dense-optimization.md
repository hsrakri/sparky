# Qwen3.8-27B dense on a DGX Spark — the roofline, and what actually speeds it up

> Day-one recipe for serving **`unsloth/Qwen3.8-27B-NVFP4`** on one GB10 as the **coder** in an orchestrator→coder agent stack — co-located with Nemotron-3.5-Lightning ([Recipe 01](../README.md)). Every number below was measured on this box.

> ### ⚡ Correction (2026-08-14): MTP is the answer — and it was in the checkpoint all along
> This recipe originally concluded "no MTP (that was the MoE's trick)" and settled for no speculative decoding. **That was wrong.** The `unsloth/Qwen3.8-27B-NVFP4` checkpoint **ships a trained MTP (multi-token-prediction) head** — 15 `mtp.*` tensors, arch `Qwen3_5ForConditionalGeneration`, `mtp_num_hidden_layers=1` — and our existing **`vllm/vllm-openai:v0.26.0-aarch64`** image already registers the `Qwen3_5MTP` architecture. We just never flipped the flag.
>
> Enabling MTP speculative decoding at depth 3 takes the coder **11.6 → 30.4 tok/s (2.6×)** — **and tool-calling stays clean**, because it's a real trained draft head, not n-gram pattern matching. This is the config we now run in production.
>
> Hat-tip to **[@drowzeys](https://github.com/drowzeys/keys-vLLm.0.27-Qwen3.8-NVFP4-MTP3-Single-DGX-Spark)** for surfacing the MTP-on-Spark result (measured 31.7 tok/s on the vLLM 0.27 `eugr/spark-vllm-b12x` GB10 build), and to **[@eugr](https://hub.docker.com/u/eugr)** for the GB10 vLLM work. Our addition: **you don't need the 0.27 nightly** — the v0.26 aarch64 image already drives the MTP head.

## TL;DR

| Config | Long-code decode | Tool-calling |
|---|---|---|
| Baseline (CUDA graphs, no spec) | **11.6–11.8 tok/s** | ✅ clean |
| + n-gram spec-decode k=5 | 22.1 tok/s (1.9×) | ❌ **mangled** |
| **+ MTP spec-decode n=3** | **30.4 tok/s (2.6×)** | ✅ **clean** ← **champion** |

**The punchline:** the checkpoint's **built-in MTP head** nearly triples decode **with clean tool-calls** — the right lever for an agent coder. The n-gram trick also speeds decode but **breaks `qwen3_coder` tool-call parsing** on this build (function name absorbs the arguments, `arguments` comes back empty), so it's disqualifying for tool serving. Use MTP.

## Why a dense 27B is "slow" here — the roofline

Decode is memory-bandwidth-bound: every generated token streams all active weights through the memory bus.

```
GB10 bandwidth        ≈ 273 GB/s
NVFP4 27B weights     ≈ 14 GB active per token
ceiling               ≈ 273 / 14 ≈ ~20 tok/s
measured baseline     = 11.6 tok/s  (~58% of ceiling, co-located at 0.45)
```

Compare the previous coder, Qwen3.6-35B-**A3B** (MoE, 3B active): **114 tok/s** — because only ~1.5 GB of experts fire per token. Dense-vs-MoE is a **~10× decode gap by design**, not a config problem. You pay it for the quality of all 27B params on every token.

> **Arch correction:** this "27B" is not a plain dense transformer — its config reports `Qwen3_5ForConditionalGeneration`, a **hybrid model with linear-attention layers, some MoE MLP layers, a vision tower, and an MTP head**. The roofline intuition (decode streams the active weights) still holds and predicts the right order of magnitude, but "dense 27B" oversimplifies it.

## What we tried, with real numbers

### 1. MTP speculative decoding — the real lever ✅ (champion)
The checkpoint ships a **trained MTP draft head** (`mtp.*` tensors), and `vllm/vllm-openai:v0.26.0-aarch64` registers `Qwen3_5MTP`. vLLM shares the target model's embedding + `lm_head` with the draft head and proposes `n` tokens per step:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Measured single-stream (400-token codegen, co-located @0.45, temp 0):
- **30.4 tok/s (2.6× the 11.6 baseline)**, and the boot log confirms `Detected MTP model. Sharing target model embedding/lm_head weights with the draft model.`
- **Tool-calling stays CLEAN** — `get_weather` with `{"city":"Tokyo","unit":"celsius"}` parsed correctly. This is the decisive difference from n-gram: a trained head drafts coherent structured tokens, so the `qwen3_coder` parser doesn't mis-split.
- Depth is effectively capped at 3 (one MTP layer); acceptance collapses beyond, matching @drowzeys's finding.

### 2. n-gram (prompt-lookup) speculative decoding — fast but disqualifying for tools
No draft model; vLLM drafts by matching n-grams in the context. Measured on a 900-token generation: **k=5 → 22.1 tok/s** (accept length 3.55), **k=3 → 19.1 tok/s**.

⚠️ **Why we don't use it:** with n-gram spec-decode enabled (either k), tool calls came back as `name="get_weather(city=\"Tokyo\", unit=\"celsius\")"` with **empty `arguments`** — the `qwen3_coder` parser mis-splits under speculative streaming. MTP does **not** have this problem, so MTP wins outright for agent serving. (n-gram is fine for pure codegen with no tool calls.)

### 3. Prefix caching — free win for agent workloads
```bash
--enable-prefix-caching
```
Agent stacks (Hermes here) resend a large system prompt + tool schema every call; prefix caching skips that prefill. Costs nothing, keep it always.

### 4. What doesn't move the needle
- Swapping NVFP4 builds (unsloth vs the gated nvidia modelopt) — same weights-per-token, same roofline.
- More `--gpu-memory-utilization` — decode isn't KV-starved; it's bandwidth-starved. (MTP is what breaks the naive per-load ceiling, by verifying >1 token per weight-load.)
- Chasing the vLLM 0.27 GB10 nightly for MTP — **unnecessary**; the v0.26 aarch64 image already has the `Qwen3_5MTP` arch.

## Final recommended launch (agent/tool serving) — with MTP

```bash
docker run -d --name qwen38-vllm --restart no \
  --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:v0.26.0-aarch64 \
  unsloth/Qwen3.8-27B-NVFP4 \
    --attention-backend flashinfer \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-model-len 131072 --max-num-seqs 32 \
    --enable-prefix-caching \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --gpu-memory-utilization 0.45 \
    --api-key "$KEY" --host 0.0.0.0 --port 8000
```

The one added line — `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` — is the whole 2.6× win. It composes cleanly with `--enable-prefix-caching` and tool-calling.

Notes:
- **compressed-tensors NVFP4**: no `--quantization` flag (auto-detected). The MTP head loads automatically from the checkpoint.
- `--max-num-seqs 32` is required at util 0.45 (GDN/Mamba cache-block sizing fails at the default 256).
- Co-located with Nemotron-3.5 (:8001) per [Recipe 01](../README.md)'s ordering rules — coder up first. At the co-located util the MTP coder + orchestrator both stay resident.
- Cold start ~7 min. Boot log should show `Detected MTP model` — if it doesn't, the flag didn't take.

## The swap context (Qwen3.6 → 3.8)

We replaced the Qwen3.6-35B-A3B MoE coder with this Qwen3.8-27B the morning it released — a 3-line Hermes repoint (delegation model + provider model + alias), orchestrator untouched. The old MoE weights stay cached for instant rollback/AB tests. With **MTP the decode gap to the old MoE coder shrinks from ~10× to ~3.75×** (30.4 vs 114 tok/s) — and in an orchestrator→coder pattern the coder's quality dominates anyway, since the orchestrator fronts the conversation and the coder runs in the background. MTP makes that tradeoff far easier to live with.

## Files
- [`../scripts/vllm-stack-up.sh`](../scripts/vllm-stack-up.sh) — ordered boot (coder → orchestrator)
- Recipe 01: [co-locating 2× 30B on one GB10](../README.md)
