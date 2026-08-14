# Nemotron-3.5 (orchestrator) + Qwen3.6 (coder) — co-located on 1× DGX Spark

> A vLLM recipe for running **two NVFP4 MoE models side-by-side on a single NVIDIA DGX Spark (GB10, ~121.7 GiB unified)** and wiring them into an agent as an **orchestrator → coder** pair.
>
> - **Orchestrator / harness:** `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` on `:8001`
> - **Coder (delegation target):** `nvidia/Qwen3.6-35B-A3B-NVFP4` on `:8000` — swappable to **Qwen3.8-27B**
> - Agent front-end: **Hermes** (main model = Nemotron, `delegation.*` = Qwen)
>
> By **hsrakri**. Built and measured on a real home-lab DGX Spark. Everything below is what actually worked — including the four ways it *didn't* first.

```bash
git clone https://github.com/hsrakri/sparky.git
```

---

## TL;DR

Two `vllm serve` containers on one GB10. The whole fight is memory + startup order:

- `--gpu-memory-utilization` is a **per-process share of the WHOLE pool** and must fit in **free** memory at startup → the two fractions must **sum to ≲ 0.88** and you must start them **sequentially** (coder first).
- Nemotron (hybrid Mamba-MoE) needs **`--enforce-eager`** when co-located — the CUDA-graph/torch.compile path **deadlocks** at boot otherwise.
- Don't pass `--mamba-cache-mode align` (forces block_size 4176 > default 2048 → assertion crash-loop).
- Make reboot deterministic with an **ordered boot script** + `--restart no` (no Docker boot race).

Result: both models resident at **~115/121 GiB**, both pass OpenAI tool-calling, orchestrator delegates coding to the coder.

## Hardware

| | |
|---|---|
| Node | 1× NVIDIA DGX Spark (GB10, `sm_121`, aarch64) |
| Memory | ~121.7 GiB **unified** (CPU+GPU shared) |
| Image | `vllm/vllm-openai:v0.26.0-aarch64` |
| Models | Nemotron-3.5-Lightning-30B-A3B-NVFP4 · Qwen3.6-35B-A3B-NVFP4 (both ModelOpt NVFP4) |

## Architecture

```
                 Hermes agent
                      │  main model
                      ▼
      ┌──────────────────────────────┐        delegation.*
      │  Nemotron-3.5-Lightning-30B   │ ───────────────────────┐
      │  :8001  (orchestrator/harness)│                        ▼
      │  reasoning + tool-calling     │        ┌──────────────────────────────┐
      └──────────────────────────────┘        │  Qwen3.6-35B-A3B  :8000 (coder)│
              gpu-mem 0.42 (enforce-eager)     │  → swap to Qwen3.8-27B          │
                                               └──────────────────────────────┘
                                                       gpu-mem 0.45
```

---

## The co-location rules (read this before anything else)

On a single GB10, `--gpu-memory-utilization X` means *"reserve `X × total` for THIS process, and that much must be free right now."* It is **not** a device-wide ceiling and it does **not** auto-partition between processes.

1. **Fractions must sum to ≲ 0.88.** Leave ~12 GiB for the OS. We run **Qwen 0.45 + Nemotron 0.42**.
2. **Start sequentially, coder first.** Bring Qwen up and *wait until ready*, then start Nemotron. Starting both at once races — whoever profiles second sees the other's memory as "used" and dies with `No available memory for the cache blocks`.
3. **Nemotron needs `--enforce-eager`.** Co-located, its torch.compile + CUDA-graph capture hangs at startup (weights never load, memory stays flat). Eager mode skips capture and boots reliably. Cost: lower decode throughput; graphs + DSpark speculative decoding are a *future* optimization (likely needs Nemotron running solo).
4. **No `--mamba-cache-mode align`.** It forces attention `block_size=4176`, which must be `≤ max_num_batched_tokens` (default 2048) → `AssertionError` crash-loop. Drop it (and `--enable-prefix-caching`), or raise `--max-num-batched-tokens ≥ 4176`.

---

## Quickstart

> Requires Docker with `--gpus all`, the image pulled, and the model weights in `~/.cache/huggingface` (both repos are public NVFP4 — `hf download <repo>`). Replace `$KEY` with your own API key.

**1) Coder first — Qwen3.6 on :8000 at 0.45**
```bash
docker run -d --name qwen36-vllm --restart no \
  --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:v0.26.0-aarch64 \
  nvidia/Qwen3.6-35B-A3B-NVFP4 \
    --quantization modelopt --attention-backend flashinfer --moe-backend marlin \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --gpu-memory-utilization 0.45 --api-key "$KEY" --host 0.0.0.0 --port 8000

# WAIT until ready before step 2:
until curl -fsS -H "Authorization: Bearer $KEY" http://localhost:8000/v1/models >/dev/null; do sleep 5; done
```

**2) Orchestrator second — Nemotron on :8001 at 0.42, eager**
```bash
docker run -d --name nemotron-vllm --restart no \
  --gpus all --ipc=host -p 8001:8001 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:v0.26.0-aarch64 \
  nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
    --moe-backend marlin --kv-cache-dtype fp8 --mamba-backend flashinfer \
    --enforce-eager \
    --reasoning-parser nemotron_v3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --gpu-memory-utilization 0.42 --max-model-len 131072 \
    --api-key "$KEY" --host 0.0.0.0 --port 8001
```

Cold start is ~6 min per model (weight load + JIT). NVFP4 quant is auto-detected (ModelOpt) — no `--quantization` needed for Nemotron.

## Deterministic reboot ordering

Two containers with `--restart unless-stopped` **race on boot** and reproduce the memory crash. Fix:

1. `docker update --restart no qwen36-vllm nemotron-vllm` — stop Docker from auto-starting them.
2. Drive boot from one ordered entry point: [`scripts/vllm-stack-up.sh`](scripts/vllm-stack-up.sh) (start Qwen → wait `:8000` ready → start Nemotron → wait `:8001` ready).
3. `@reboot` cron:
   ```cron
   @reboot /home/USER/.local/bin/vllm-stack-up.sh >> /home/USER/.local/state/vllm-model/stack.log 2>&1
   ```

Tradeoff: `--restart no` also disables *runtime* crash auto-recovery — deliberate, because uncoordinated auto-restart is what caused the race. Add a watchdog that re-runs the ordered script if you want recovery.

## Hermes wiring (orchestrator → coder)

```yaml
model:                       # main = orchestrator
  default: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
  provider: custom:kpopsparky-nemotron
  base_url: http://<spark>:8001/v1
delegation:                  # sub-agents = coder
  model: nvidia/Qwen3.6-35B-A3B-NVFP4
  provider: custom:kpopsparky-vllm
  base_url: http://<spark>:8000/v1
custom_providers:
  - {name: kpopsparky-nemotron, model: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4, base_url: http://<spark>:8001/v1, key_env: OPENAI_API_KEY}
  - {name: kpopsparky-vllm,     model: nvidia/Qwen3.6-35B-A3B-NVFP4,                        base_url: http://<spark>:8000/v1, key_env: OPENAI_API_KEY}
```

### Swapping the coder to Qwen3.8-27B
When Qwen3.8-27B is served on `:8000` (replacing the 3.6 container), repoint **3 spots** to the new model id (base_url unchanged): `custom_providers[kpopsparky-vllm].model`, `delegation.model`, and your coder alias. Restart the gateway. The orchestrator (Nemotron, `:8001`) is untouched.

## Benchmarks

Single-stream, temperature 0, localhost, measured on this box:

| Model | Port / util | Decode | TTFT | Tool call | Notes |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | :8000 / 0.45 (MTP+graphs) | **105–142 tok/s** | 188–228 ms | ✅ correct args | `enable_thinking:false` |
| Nemotron-3.5-Lightning-30B | :8001 / 0.42 (**eager**) | **~50 tok/s** | reasoning-gated | ✅ correct args | reasons ~128–169 tok even for a 1-word answer |

Nemotron KV cache pool: **31.4 GiB ≈ 9.56 M tokens** (fp8), 72.97× concurrency @ 131 K ctx. Both resident → system at ~115/121 GiB.

**Overnight delegation soak:** 150 back-to-back delegated coding tasks, **97.3% pass, 100% delegation, 0 infra failures, $0** → [`benchmarks/overnight-soak-2026-08-12/`](benchmarks/overnight-soak-2026-08-12/). **Correction stress:** 75 multi-turn cycles → [`benchmarks/stress-corrections/`](benchmarks/stress-corrections/).

> **Update 2026-08-14:** the coder on `:8000` is now **Qwen3.8-27B (dense NVFP4)** — see **[Recipe 02: dense-27B optimization on GB10](recipes/qwen38-27b-dense-optimization.md)** (roofline math, n-gram spec-decode 1.9× + its tool-calling gotcha, final config). Repo index: [`index.html`](index.html).

> ⚠️ Not apples-to-apples: Nemotron runs eager (no graphs/spec-decode) and is a reasoning model with thinking on; Qwen has thinking disabled. Nemotron's 50 tok/s is a floor, not its ceiling.

## Verify

```bash
curl -s -H "Authorization: Bearer $KEY" http://<spark>:8000/v1/models | jq .data[0].id
curl -s -H "Authorization: Bearer $KEY" http://<spark>:8001/v1/models | jq .data[0].id
# tool-call smoke test: expect finish_reason=tool_calls
```

## Troubleshooting — the four failure modes we hit

| Symptom | Cause | Fix |
|---|---|---|
| `No available memory for the cache blocks` | both started at same util simultaneously (race) | start **sequentially**, coder first; utils sum ≲ 0.88 |
| `Free memory on device ... less than desired GPU memory utilization (0.9, 109 GiB)` | set 2nd model's util too high (it's a per-process share, not a ceiling) | size util to fit **free** memory (~0.42 here) |
| `AssertionError: block_size (4176) must be <= max_num_batched_tokens (2048)` | `--mamba-cache-mode align` | drop it, or `--max-num-batched-tokens ≥ 4176` |
| Boots, prints banner, then **hangs before loading weights** (memory flat) | CUDA-graph/torch.compile deadlock under co-location | `--enforce-eager` |

## Credits / License

Recipe & measurements: **hsrakri** — built in the home lab with **krishavh**. Upstream: [vLLM](https://github.com/vllm-project/vllm), NVIDIA Nemotron, Qwen, [Hermes Agent](https://github.com/NousResearch/hermes-agent).

Licensed **MIT** — see [LICENSE](LICENSE). This repo is the deployment recipe (docs + scripts) only; the model **weights** are under their respective upstream licenses (NVIDIA Open Model License for Nemotron, Apache-2.0 for Qwen) and are **not** redistributed here.
