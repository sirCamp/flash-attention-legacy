#!/bin/bash
source /workspace/env.sh >/dev/null 2>&1
cd /workspace/analysis/models
while ! grep -q ALLDONE logs/run_all.log; do sleep 5; done
echo "== shadow gemma3"; python shadow.py google/gemma-3-1b-it --n 26,1000 > logs/shadow_gemma3.log 2>&1; echo exit $?
echo "== shadow gemma2"; python shadow.py google/gemma-2-2b-it --n 26,1000 > logs/shadow_gemma2.log 2>&1; echo exit $?
echo "== sinks"; python sinks_probe.py > logs/sinks_probe.log 2>&1; echo exit $?
echo "== bert"; python encoders.py google-bert/bert-base-uncased --tag bert_base --n 512 --gold > logs/bert_base.log 2>&1; echo exit $?
echo "== modernbert"; python encoders.py answerdotai/ModernBERT-base --tag modernbert_base --n 1000 --gold > logs/modernbert_base.log 2>&1; echo exit $?
echo "== qwen3emb"; python encoders.py Qwen/Qwen3-Embedding-0.6B --tag qwen3_embedding --n 1000 --pool last --gold > logs/qwen3_embedding.log 2>&1; echo exit $?
echo "== train"; python train_probe.py > logs/train_probe.log 2>&1; echo exit $?
echo ALLDONE2
