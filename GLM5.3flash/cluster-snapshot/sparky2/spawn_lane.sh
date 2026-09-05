#!/bin/bash
# spawn the coding lane on this box, fully detached
pkill -f "lane_proxy[.]py" 2>/dev/null
sleep 1
setsid nohup python3 /home/haarithd/lane_proxy.py 8892 coding 10 high http://10.100.128.1:8888 \
  >> /home/haarithd/lane-coding.log 2>&1 < /dev/null &
sleep 2
echo "listening: $(ss -ltn | grep -c 8892)"
tail -1 /home/haarithd/lane-coding.log 2>/dev/null
