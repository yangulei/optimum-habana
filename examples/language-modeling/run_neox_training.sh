res_dir=res-clm-gpt-neox-20b-zero1-offload-bs8-ddp4
mkdir -p ${res_dir}

PT_HPU_POOL_MEM_ACQUIRE_PERC=100 \
python ../gaudi_spawn.py \
    --world_size 4 --use_deepspeed run_clm.py \
    --model_name_or_path EleutherAI/gpt-neox-20b \
    --cache_dir /hf_cache/ \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --block_size 1024 \
    --per_device_train_batch_size 8 \
    --do_train \
    --do_eval False \
    --output_dir ${res_dir} \
    --overwrite_output_dir \
    --gaudi_config_name Habana/gpt2 \
    --use_habana \
    --use_lazy_mode \
    --gradient_checkpointing \
    --throughput_warmup_steps 3 \
    --save_strategy  no \
    --logging_steps 10 \
    --max_steps 20 \
    --deepspeed ds_config_zero_1_offload.json \
    2>&1 \
    | tee ${res_dir}/run.log
rm ${res_dir}/pytorch_model*
