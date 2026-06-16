#!/usr/bin/env bash
# Safe restart: ONE pipeline instance on 6 cameras spread across all 4 sites.
# This is the proven-stable single-process path (serialize_gpu) — NOT the 4-shard
# concurrent-GPU run that correlated with the server crash. Watch dmesg for GPU
# faults in another terminal:   dmesg -w | grep -i xid
set -uo pipefail
cd /home/cv-gpu-2/harshit_workspace/bangalore_demo
source /home/cv-gpu-2/harshit_workspace/.venv/bin/activate

# 6 cameras, one or two per IP location (cam01:site1, cam04/05:site2, cam08/09:site3, cam12:site4)
CAMS="cam01,cam04,cam05,cam08,cam09,cam12"

rm -f /tmp/demo6.log
nohup python -W ignore run.py --only "${CAMS}" > /tmp/demo6.log 2>&1 &
echo $! > /tmp/demo6.pid
echo "started pid $(cat /tmp/demo6.pid) on cameras: ${CAMS}"
echo "watch:  tail -f /tmp/demo6.log | grep --line-buffered -E 'ODN:|DeepSort:|VIOLATION:|LPR:|\[hb\]|GPU fault'"
echo "stop :  kill \$(cat /tmp/demo6.pid)"
