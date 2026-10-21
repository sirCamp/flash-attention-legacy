#!/bin/bash
source /workspace/env.sh >/dev/null 2>&1
cd /workspace/analysis/models
while ! grep -q ALLDONE4 logs/run_all4.log; do sleep 5; done
echo "== train2"; python train_probe2.py > logs/train_probe2.log 2>&1; echo exit $?
echo ALLDONE5
