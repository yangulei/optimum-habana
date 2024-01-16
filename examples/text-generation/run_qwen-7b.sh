python run_generation.py \
    --model_name_or_path /workspace/models/qwen-7b-chat \
    --bf16 \
    --trust_remote_code \
    --use_hpu_graphs \
    --use_kv_cache \
    --max_new_tokens 100 \
    --prompt "How are you today?"