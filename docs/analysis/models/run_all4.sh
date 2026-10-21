#!/bin/bash
source /workspace/env.sh >/dev/null 2>&1
cd /workspace/analysis/models
while ! grep -q ALLDONE3 logs/run_all3.log; do sleep 5; done
echo "== shadow bert"; python shadow.py google-bert/bert-base-uncased --encoder --n 64,512 > logs/shadow_bert.log 2>&1; echo exit $?
echo "== shadow modernbert"; python shadow.py answerdotai/ModernBERT-base --encoder --n 64,1000 > logs/shadow_modernbert.log 2>&1; echo exit $?
echo "== shadow qwen25"; python shadow.py Qwen/Qwen2.5-0.5B-Instruct --n 26,1000 > logs/shadow_qwen25.log 2>&1; echo exit $?
echo ALLDONE4
