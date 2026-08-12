# Overnight delegation soak — 2026-08-12

150 back-to-back coding tasks driven through **Hermes**, with **Nemotron-3.5-Lightning-30B** (`:8001`) orchestrating and delegating every implementation to **Qwen3.6-35B-A3B** (`:8000`) as the coder. One DGX Spark (GB10). 100% local inference — **$0 API cost**.

## Results

| Metric | Value |
|---|---|
| Tasks | **150 / 150** completed |
| Pass rate | **146 OK / 4 fail — 97.3%** |
| Delegation | **150 / 150 → 100%** routed orchestrator → coder |
| Infra failures / crashes / auto-recoveries | **0 / 0 / 0** |
| LLM calls (all local) | **~1,348** — 819 Qwen (coder) + 529 Nemotron (orchestrator) |
| Wall per task | avg **104s** · min 49s · max 300s |
| Compute | ~4.3h task-time over ~5h wall · both containers stable 8h · memory flat ~119–120/121 GiB |

### Wall-time distribution
```
<60s     █████████ 17
60–90s   ██████████████████████████████ 61
90–120s  ██████████████████ 36
120–180s ██████████ 21
≥180s    ███████ 15
```

### The only failures — diagnostic, not infra
All **4 failures were identical**: exit 124 (hit the 300s per-task cap) because the delegated **coder over-iterated** — 17–30 Qwen calls instead of the usual ~3. Concentrated on the two hardest problems:

| prompt_idx | problem | pass |
|---|---|---|
| 13 | sliding-window-log rate limiter | 3 / 6 |
| 9 | binary-tree serialize/deserialize | 5 / 6 |

The other **22 of 24** problem types passed **100%**. Never a memory, endpoint, or crash failure — the self-healing net (ordered restart on endpoint drop) never had to fire.

### Takeaway
The **orchestrator → coder** design is stable at scale: 150 sequential delegations overnight, zero infra failures, on a single GB10. The lone weakness is *model behavior* (coder looping on hard tasks past the timeout), not the stack — an easy future tweak is a sub-agent iteration cap or a longer per-task budget for hard problems.

## Files
- [`results.csv`](results.csv) — one row per task: `iter,ts,prompt_idx,wall_s,exit,qwen_delta,nemo_delta,delegated,status`
  - `qwen_delta` = coder (Qwen :8000) request count for that task — >0 confirms delegation fired.

## Method
Loop harness: `scripts/vllm-stack-up.sh` kept the stack ordered; each iteration health-checked both endpoints (auto-recover via the ordered boot script), ran one delegated coding task via `hermes -z`, and recorded per-endpoint request deltas. 24 rotating problems (data structures, decorators, rate limiters, parsers). temperature 0, 300s per-task timeout, ~20s between tasks.
