#!/bin/bash
source /workspace/env.sh >/dev/null 2>&1
cd /workspace/analysis/models
run() { tag=$1; shift; echo "=== $tag $(date)"; python survey.py "$@" --tag $tag > logs/$tag.log 2>&1; echo "exit $? $tag"; }
run qwen25_05b Qwen/Qwen2.5-0.5B-Instruct --gold
run qwen3_06b Qwen/Qwen3-0.6B --gold
run gemma3_1b google/gemma-3-1b-it --gold --long 5000
run gemma2_2b google/gemma-2-2b-it --gold --long 5000
run phi3_mini microsoft/Phi-3-mini-4k-instruct --gold --long 3500
run mistral7b_v03 mistralai/Mistral-7B-Instruct-v0.3
run mistral7b_v01 mistralai/Mistral-7B-v0.1 --long 5000
echo ALLDONE
