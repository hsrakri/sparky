# GLM-5.3-Flash 2× DGX Spark — cluster snapshot (decommissioned 2026-09-05)

Everything needed to bring the TP=2 GLM-5.3-Flash EXL3 cluster back exactly as it last ran.
Model weights (`brandonmusic/GLM-5.3-Flash-tr3-4bpw`, 164 GB) and the DFlash2 drafter
(`incoai/GLM-5.3-Flash-DFlash2`) were left cached in `~/.cache/huggingface` on BOTH Sparks.

| file | what |
|---|---|
| `sparky1/.env` | the kit's env: model, MNBT=3584 (page-aligned), spinwait 16 ms, priority scheduling |
| `sparky1/start.sh.patch` | one-line local patch to MiaAI-Lab's `start.sh` (passes prefix-cache retention env), plus kit commit + image id |
| `sparky1/launch-dflash2-tp2.sh` | DFlash2 k=7 speculative-decoding launcher |
| `sparky1/lane_proxy.py`, `spawn_agent.sh`, `sparky2/spawn_lane.sh` | the two-lane QoS proxies (:8891 agent / :8892 coding) |
| `sparky1/set_governor.sh`, `cap_freq.sh` | thermal levers (schedutil + 3.3 GHz cap) — run via privileged busybox, see the story page |
| `*/docker-inspect-*.json` | full container config as last run (API key + LAN IPs redacted) |
| `sparky1/engine-final.log` | last 400 lines incl. DFlash2 acceptance metrics |

Return path: restore `.env`, apply the patch, `SKIP_DOWNLOAD=1 bash ./start.sh` on sparky1, then
`spawn_agent.sh` / `spawn_lane.sh`. ~8 min to first token.
