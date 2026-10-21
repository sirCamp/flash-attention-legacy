#!/bin/bash
source /workspace/env.sh >/dev/null 2>&1
cd /workspace/analysis/models
while ! grep -q ALLDONE2 logs/run_all2.log; do sleep 5; done
echo "== timing qwen25"; python timing.py Qwen/Qwen2.5-0.5B-Instruct qwen25 > logs/timing_qwen25.log 2>&1; echo exit $?
echo "== timing qwen3"; python timing.py Qwen/Qwen3-0.6B qwen3 > logs/timing_qwen3.log 2>&1; echo exit $?
echo "== timing mistral"; python timing.py mistralai/Mistral-7B-Instruct-v0.3 mistral7b_v03 512,1024,2048,4096,8192 > logs/timing_mistral.log 2>&1; echo exit $?
echo ALLDONE3
