# python ../gaudi_spawn.py --use_deepspeed --world_size 8 run_generation.py \
GRAPH_VISUALIZATION=1 \
PT_HPU_POOL_MEM_ACQUIRE_PERC=100 \
python run_generation.py \
--model_name_or_path /models/internlm-20b \
--max_new_tokens 100 \
--bf16 \
--use_hpu_graphs \
--use_kv_cache \
--batch_size 8 \
--attn_softmax_bf16 \
--limit_hpu_graphs \
--reuse_cache \
--trim_logits \
--prompt "Hello world"