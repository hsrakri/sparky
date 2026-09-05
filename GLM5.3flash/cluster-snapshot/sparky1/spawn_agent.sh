#!/bin/bash
pgrep -f "lane_proxy.py 8891" >/dev/null && { echo "already up"; exit 0; }
setsid nohup python3 /home/haarithd/lane_proxy.py 8891 agent 0 low http://localhost:8888 >>/home/haarithd/lane-agent.log 2>&1 </dev/null &
sleep 2; ss -ltn | grep -q 8891 && echo "respawned 8891" || echo "FAILED 8891"
