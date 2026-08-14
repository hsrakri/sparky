# Qwen3.8-27B dense on a DGX Spark — the roofline, and what actually speeds it up

> Day-one recipe for serving **`unsloth/Qwen3.8-27B-NVFP4`** (dense 27B, Unsloth Dynamic V3.0 NVFP4) on one GB10 as the **coder** in an orchestrator→coder agent stack — co-located with Nemotron-3.5-Lightning ([Recipe 01](../README.md)). Every number below was measured on this box the day the model dropped.

## TL;DR

| Config | Long-code decode | Tool-calling |
|---|---|---|
| Baseline (dense, CUDA graphs) | **11.6–11.8 tok/s** | ✅ clean |
| + n-gram spec-decode k=5 | **22.1 tok/s (1.9×)** | ❌ **mangled** |
| + n-gram spec-decode k=3 + prefix cache | **19.1 tok/s (1.65×)** | ❌ **mangled** |
| **Final: prefix cache, no spec-decode** | ~11.8 tok/s | ✅ clean |

**The punchline:** n-gram speculative decoding nearly doubles dense-27B code generation with **no draft model and no quality loss on plain text** — but on this vLLM build (`v0.26.0`) it **breaks `qwen3_coder` tool-call parsing** (function name absorbs the arguments, `arguments` comes back empty). For an agent coder that writes files via tool calls, that's disqualifying. Pure-codegen serving? Turn it on. Agent/tool serving? Leave it off until the parser interaction is fixed.

## Why a dense 27B is "slow" here — the roofline

Decode is memory-bandwidth-bound: every generated token streams all active weights through the memory bus.

```
GB10 bandwidth        ≈ 273 GB/s
NVFP4 27B weights     ≈ 14 GB active per token
ceiling               ≈ 273 / 14 ≈ ~20 tok/s
measured baseline     = 11.6 tok/s  (~58% of ceiling, co-located at 0.45)
```

Compare the previous coder, Qwen3.6-35B-**A3B** (MoE, 3B active): **114 tok/s** — because only ~1.5 GB of experts fire per token. Dense-vs-MoE is a **~10× decode gap by design**, not a config problem. You pay it for the quality of all 27B params on every token.

## What we tried, with real numbers

### 1. n-gram (prompt-lookup) speculative decoding — the big lever
No draft model needed: vLLM drafts continuation tokens by matching n-grams already in the context — and code is extremely self-similar (`assert`, `def`, repeated identifiers).

```bash
--speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'
```

Measured on a 900-token class-plus-15-tests generation:
- **k=5:** 22.1 tok/s; mean acceptance length 3.55; draft acceptance climbing 37% → 51% as the pattern cache warms; per-position acceptance 0.70/0.63/0.44/0.41/0.38 (positions 4–5 mostly wasted).
- **k=3:** 19.1 tok/s; acceptance length ~3.0; draft acceptance 55–67% (much less wasted draft).

⚠️ **The gotcha that decided it:** with spec-decode enabled (either k), tool calls came back as `name="get_weather(city=\"Tokyo\", unit=\"celsius\")"` with **empty `arguments`** — the `qwen3_coder` parser mis-splits under speculative streaming. Without spec-decode the same request parses perfectly. Temperature 0 both times, so this is a serving-stack interaction, not sampling noise.

### 2. Prefix caching — free win for agent workloads
```bash
--enable-prefix-caching
```
Agent stacks (Hermes here) resend a large system prompt + tool schema every call; prefix caching skips that prefill. Costs nothing, keep it always.

### 3. What doesn't move the needle
- Swapping NVFP4 builds (unsloth vs the gated nvidia modelopt) — same weights-per-token, same roofline.
- More `--gpu-memory-utilization` — decode isn't KV-starved; it's bandwidth-starved.

## Final recommended launch (agent/tool serving)

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
    --gpu-memory-utilization 0.45 \
    --api-key "$KEY" --host 0.0.0.0 --port 8000
```

For **pure code generation** (no tool calls), add the n-gram spec-decode line above for ~1.9×.

Notes:
- **Dense/compressed-tensors**: no `--quantization` flag (auto-detected), no `--moe-backend`, no MTP (that was the MoE's trick).
- `--max-num-seqs 32` is required at util 0.45 (GDN/Mamba cache-block sizing fails at the default 256).
- Co-located with Nemotron-3.5 (:8001, util 0.42) per [Recipe 01](../README.md)'s ordering rules — coder up first. System sits ~117/121 GiB.
- Cold start ~7 min.

## The swap context (Qwen3.6 → 3.8)

We replaced the Qwen3.6-35B-A3B MoE coder with this dense 27B the morning it released — a 3-line Hermes repoint (delegation model + provider model + alias), orchestrator untouched. The old MoE weights stay cached for instant rollback/AB tests. Tradeoff accepted: **10× slower decode for a materially stronger coder** — in an orchestrator→coder pattern the coder's quality dominates, since the fast orchestrator (~50 tok/s) fronts the conversation and the coder runs in the background.

## Files
- [`../scripts/vllm-stack-up.sh`](../scripts/vllm-stack-up.sh) — ordered boot (coder → orchestrator)
- Recipe 01: [co-locating 2× 30B on one GB10](../README.md)
