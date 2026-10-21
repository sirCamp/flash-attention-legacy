from huggingface_hub import snapshot_download
ms=["Qwen/Qwen2.5-0.5B-Instruct","Qwen/Qwen3-0.6B","google/gemma-3-1b-it","google/gemma-2-2b-it","microsoft/Phi-3-mini-4k-instruct","answerdotai/ModernBERT-base","google-bert/bert-base-uncased","Qwen/Qwen3-Embedding-0.6B","mistralai/Mistral-7B-Instruct-v0.3","mistralai/Mistral-7B-v0.1"]
for m in ms:
    try:
        p=snapshot_download(m, allow_patterns=["*.json","*.safetensors","*.model","*.txt","*.py","tokenizer*","*.tiktoken","1_Pooling/*","modules.json"], ignore_patterns=["consolidated*","original/*","*.pth","onnx/*"])
        print("done",m,flush=True)
    except Exception as e:
        print("FAIL",m,type(e).__name__,str(e)[:200],flush=True)
