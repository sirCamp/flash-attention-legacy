# Task accuracy with each attention implementation

`python benchmarks/tasks.py --model meta-llama/Llama-3.1-8B-Instruct --passkey-lens 4096 8192 16384 32768 65536 --save`
(raw results with every document's prediction: `tasks_<gpu>.jsonl`; these tables: `--report`). Tesla V100 32 GB,
transformers 5.20.2, torch 2.7.1, lm-evaluation-harness 0.4.13, fp16 weights, kernels at commit 5ad45ca. The same model is loaded
with each implementation in its own process and handed to lm-eval: same documents, prompts and few-shot examples (lm-eval's
defaults: ARC and HellaSwag 0-shot by log-likelihood, GSM8K 5-shot greedy generation of up to 256 tokens in batches of 16 with
left padding, so the prefill and the decode kernels are both exercised).

- arc_easy (2376), arc_challenge (1172), hellaswag (10042): acc and acc_norm. GSM8K (1319): exact match with lm-eval's strict
  and flexible answer extraction. Standard errors (eager): ARC-Easy 0.008, ARC-Challenge 0.015, HellaSwag 0.004-0.005, GSM8K
  0.012.
- passkey_N: a 5-digit key hidden at depth (i + 0.5) / 20 in N tokens of filler (the prompt is a few tokens shorter than N),
  greedy answer with the chat template; accuracy over 20 keys.
- Each cell: the score, and in parentheses how many documents' answers differ from eager's (the chosen option, or the extracted
  answer). Datasets: allenai/ai2_arc (CC BY-SA 4.0), Rowan/hellaswag (MIT), openai/gsm8k (MIT).

Reading the numbers: every score agrees within a fraction of its standard error; ours changes about as many answers as SDPA
does (near-ties in fp16 logits: 2-21 documents per task, in both directions), and where SDPA and eager run out of memory (8k+
tokens on a 32 GB V100 with an 8B model) the kernel keeps retrieving every key up to 32k. At 64k the KV cache (8 GiB) and the
weights (15 GiB) leave no room for any implementation.

#### Tesla V100-SXM2-32GB: meta-llama/Llama-3.1-8B-Instruct (fp16; score, answers different from eager)

| task | metric | n | flash_attn_legacy | sdpa | eager |
|---|---|---|---|---|---|
| arc_easy | acc | 2376 | 0.8220 (7) | 0.8215 (4) | 0.8224 |
| arc_easy | acc_norm | 2376 | 0.7971 (7) | 0.7980 (4) | 0.7984 |
| arc_challenge | acc | 1172 | 0.5375 (2) | 0.5375 (2) | 0.5384 |
| arc_challenge | acc_norm | 1172 | 0.5589 (2) | 0.5589 (2) | 0.5589 |
| hellaswag | acc | 10042 | 0.5977 (6) | 0.5977 (8) | 0.5978 |
| hellaswag | acc_norm | 10042 | 0.7950 (6) | 0.7953 (8) | 0.7953 |
| gsm8k | exact_match (strict-match) | 1319 | 0.7119 (18) | 0.7111 (20) | 0.7149 |
| gsm8k | exact_match (flexible-extract) | 1319 | 0.7733 (21) | 0.7741 (21) | 0.7756 |
| passkey_4096 | acc | 20 | 1.0000 (0) | 1.0000 (0) | 1.0000 |
| passkey_8192 | acc | 20 | 1.0000 | OOM | OOM |
| passkey_16384 | acc | 20 | 1.0000 | OOM | OOM |
| passkey_32768 | acc | 20 | 1.0000 | OOM | OOM |
| passkey_65536 | acc | 20 | OOM | OOM | OOM |
