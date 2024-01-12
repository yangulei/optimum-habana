python run_generation.py \
    --model_name_or_path /workspace/models/qwen-7b \
    --trust_remote_code \
    --use_hpu_graphs \
    --bucket_size 512 \
    --use_kv_cache \
    --max_new_tokens 100 \
    --prompt "Here is my prompt"