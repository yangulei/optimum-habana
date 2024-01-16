python ../gaudi_spawn.py --use_deepspeed --world_size 4 \
    run_generation.py \
    --bf16 \
    --model_name_or_path /workspace/models/qwen-72b-chat \
    --trust_remote_code \
    --use_hpu_graphs \
    --use_kv_cache \
    --max_new_tokens 100 \
    --prompt "How are you today?"