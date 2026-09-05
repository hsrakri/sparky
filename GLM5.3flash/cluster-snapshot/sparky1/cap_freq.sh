#!/bin/sh
CAP=$1
for p in /sys/devices/system/cpu/cpufreq/policy*/scaling_max_freq; do
  hw=$(cat ${p%scaling_max_freq}cpuinfo_max_freq)
  if [ "$hw" -gt "$CAP" ]; then echo "$CAP" > "$p"; else echo "$hw" > "$p"; fi
done
grep . /sys/devices/system/cpu/cpufreq/policy*/scaling_max_freq | head -4
