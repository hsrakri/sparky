# Multi-turn correction stress test — 2026-08-13

A harder follow-up to the [single-round overnight soak](../overnight-soak-2026-08-12/). Each task is a **two-round correction loop** through the **Nemotron→Qwen** delegation on one DGX Spark, 100% local ($0):

1. **Round 1** — the orchestrator delegates *"write intense code + assert self-tests to `solution.py`"* to the Qwen coder. The code is then **executed**.
2. **Round 2 (correction)** — a *real* correction is fed back and the code is revised:
   - if round 1 **failed** → the actual error trace ("fix this")
   - if it **passed** → a tougher requirement (type hints, edge-case asserts, complexity, thread-safety)
3. The revised `solution.py` is executed again.

24 rotating *intense* problems (Myers diff, mini-regex engine, A*, consistent-hash ring, interval tree, B-tree, min-max heap, coroutine scheduler…). 600s per-round timeout.

## Results (75 correction cycles)

| Metric | Value |
|---|---|
| Cycles | **75** |
| End-to-end OK (round 2 passes) | **59 / 75 — 79%** |
| Delegated Qwen calls | **1,785** (round 1: 1,646 · round 2: 139) |
| Round-1 delegation rate | **74/75 (99%)**, avg **21.9 calls/task** |
| Round-2 delegation rate | **11/75 (15%)** |
| Round 1 | pass 49 · **timeout 26** |
| Round 2 | pass 59 · timeout 16 |
| Wall/cycle | avg **649s** · max 1252s |
| Auto-recoveries / crashes | **0 / 0** · memory flat ~120/121 GiB |

## The finding: the orchestrator's division of labor

**Nemotron delegates the hard first draft to Qwen, then patches corrections itself.**
- Round 1 leans *hard* on the coder — **~22 delegated calls/task** to write intense code.
- Round 2 corrections are handled **inline by the orchestrator** (only 15% re-delegated) yet still land a passing `solution.py` 79% of the time.

## Failures are timeouts, not delegation breakdowns

~35% of round-1 attempts hit the 600s cap because the **coder over-iterates on genuinely hard algorithms**, never converging in time. Per-task pass rates make it obvious:

| Clean (all pass) | Struggles (coder over-iterates → timeout) |
|---|---|
| LRU-TTL (4/4), A*, topo-sort, expr-eval, PN-counter, bloom filter, circuit-breaker, consistent-hash, union-find, coroutine scheduler, Levenshtein, KMP, fixed-decimal (all 3/3) | **Myers diff (0/4!)**, min-max heap (1/3), thread-safe object pool (1/3), COW B-tree (2/3), interval tree (2/3) |

Even when round 1 times out, **round 2 usually recovers it** — hence 79% overall.

## vs. the single-round soak
[Soak #1](../overnight-soak-2026-08-12/) (single round, easy–medium tasks): **150 tasks, 97%**. This one (multi-turn corrections, *intense* tasks): **79%**. The gap is entirely the harder problems + the coder-over-iteration timeout — **not** the delegation pipeline, which stayed rock-solid (0 recoveries, 0 crashes, 1,785 clean delegated calls overnight).

## Files
- [`results.csv`](results.csv) — `iter,ts,task_idx,r1_qd,r1_run,corr,r2_qd,r2_run,wall,status`
  - `r1_qd`/`r2_qd` = Qwen (coder) request delta per round · `r*_run` ∈ {pass, fail, timeout, nocode} · `corr` ∈ {twist, error, redo}
