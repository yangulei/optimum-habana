# hl-prof-config -e off -phase=multi-enq -g 1-2000 -s train_llama_torch_0904am-g1_2000
export GRAPH_VISUALIZATION=1
# export HABANA_PROFILE=1
export LOG_LEVEL_PT_FALLBACK=1

python ../gaudi_spawn.py \
    --world_size 8 --use_deepspeed run_clm_llama.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./test-clm-llama \
    --gaudi_config_name Habana/gpt2 \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --gradient_checkpointing \
    --use_cache False \
    --throughput_warmup_steps 3 \
    --num_train_epochs 3 \
    --deepspeed ./ali_ds.json

# python ../gaudi_spawn.py \
#     --world_size 8 --use_deepspeed run_lora_clm.py \
#     --model_name_or_path huggyllama/llama-7b \
#     --dataset_name tatsu-lab/alpaca \
#     --bf16 True \
#     --output_dir ./model_lora_llama \
#     --num_train_epochs 30 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 4 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 2000 \
#     --save_total_limit 1 \
#     --learning_rate 1e-4 \
#     --logging_steps 1 \
#     --dataset_concatenation \
#     --do_train \
#     --use_habana \
#     --use_lazy_mode \
#     --throughput_warmup_steps 3 \
#     --deepspeed ./ds_config.json

# python ../gaudi_spawn.py \
#     --world_size 8 --use_deepspeed run_clm.py \
#     --model_name_or_path gpt2-xl \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-2-raw-v1 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --do_train \
#     --do_eval \
#     --learning_rate 4e-4 \
#     --output_dir /tmp/test-clm \
#     --gaudi_config_name Habana/gpt2 \
#     --use_habana \
#     --use_lazy_mode \
#     --use_hpu_graphs_for_inference \
#     --gradient_checkpointing \
#     --use_cache False \
#     --throughput_warmup_steps 3 \
#     --deepspeed path_to_my_deepspeed_config

# python ../gaudi_spawn.py \
#     --world_size 8 --use_mpi run_lora_clm.py \
#     --model_name_or_path huggyllama/llama-7b \
#     --dataset_name tatsu-lab/alpaca \
#     --bf16 True \
#     --output_dir ./model_lora_llama \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 4 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 2000 \
#     --save_total_limit 1 \
#     --learning_rate 1e-4 \
#     --logging_steps 1 \
#     --dataset_concatenation \
#     --do_train \
#     --use_habana \
#     --use_lazy_mode \
#     --throughput_warmup_steps 3