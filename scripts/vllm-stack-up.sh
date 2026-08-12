#!/usr/bin/env bash
# vllm-stack-up.sh — bring up the co-located vLLM stack in the REQUIRED order.
#
# WHY THIS EXISTS: on this single GB10 (~121.7 GiB usable) `--gpu-memory-utilization`
# is a per-process share of the WHOLE pool and must fit in FREE memory at startup.
# Qwen (coder, :8000, 0.45) MUST be up and READY before Nemotron (orchestrator,
# :8001, 0.42). Starting them together races and the 2nd dies with
#   "ValueError: No available memory for the cache blocks".
# Both containers are set to `--restart no` so Docker does NOT race-start them on
# boot; this script (via @reboot cron) is the single ordered entry point.
# Refs: ~/Desktop/AgentMemory/infrastructure.md + changelog.md (2026-08-11).
set -uo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

log(){ printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"; }

wait_docker(){
  docker info >/dev/null 2>&1 && return 0
  log "waiting for docker daemon..."
  local i; for i in $(seq 1 60); do
    sleep 5; docker info >/dev/null 2>&1 && { log "docker up after ~$((i*5))s"; return 0; }
  done
  log "ERROR: docker daemon never became available"; return 1
}

key_of(){ docker inspect "$1" --format '{{join .Args " "}}' 2>/dev/null | grep -oP -- '--api-key \K\S+'; }

wait_ready(){
  local port="$1" key="$2" i
  for i in $(seq 1 180); do
    curl -fsS -H "Authorization: Bearer $key" "http://localhost:${port}/v1/models" >/dev/null 2>&1 \
      && { log "ready on :$port after ~$((i*5))s"; return 0; }
    sleep 5
  done
  log "ERROR: :$port did not become ready within 15 min"; return 1
}

start_one(){
  local cname="$1" port="$2"
  if ! docker ps -a --format '{{.Names}}' | grep -qx "$cname"; then
    log "ERROR: container '$cname' does not exist. Recreate it first (vllm-model.sh up <profile> or the documented docker run)."
    return 1
  fi
  if docker ps --format '{{.Names}}' | grep -qx "$cname"; then
    log "$cname already running"
  else
    log "starting $cname ..."
    docker start "$cname" >/dev/null
  fi
  wait_ready "$port" "$(key_of "$cname")"
}

wait_docker || exit 1
log "=== vLLM stack up (ORDERED: Qwen coder :8000 -> Nemotron orchestrator :8001) ==="
start_one qwen36-vllm  8000 || { log "Qwen failed to come up; NOT starting Nemotron (would race)."; exit 1; }
start_one nemotron-vllm 8001 || { log "Nemotron failed to come up."; exit 1; }
log "=== stack up complete: both endpoints ready ==="
