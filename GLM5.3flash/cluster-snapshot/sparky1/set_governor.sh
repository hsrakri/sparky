#!/bin/sh
# runs INSIDE privileged container; host sysfs is writable there
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo "$1" > "$g"
done
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
