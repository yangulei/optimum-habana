res_dir=test-llama-lazy-only
mkdir -p ${res_dir}
python ../gaudi_spawn.py \
    --world_size 8 --use_deepspeed run_clm_llama.py \
    --model_name_or_path /mnt/decapoda-research/llama-7b-hf \
    --cache_dir /mnt/hf_cache/ \
    --torch_dtype bfloat16 \
    --use_cache False \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --block_size 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --learning_rate 5e-5 \
    --output_dir ${res_dir} \
    --overwrite_output_dir \
    --gaudi_config_name Habana/gpt2 \
    --use_habana \
    --use_lazy_mode True \
    --use_hpu_graphs_for_training False\
    --adjust_throughput True \
    --pipelining_fwd_bwd False \
    --non_blocking_data_copy False \
    --throughput_warmup_steps 3 \
    --save_strategy  no \
    --logging_steps 10 \
    --max_steps 200 \
    --deepspeed ds_config.json \
    2>&1 \
    | tee ${res_dir}/run.log

res_dir=test-llama-lazy-pipe
mkdir -p ${res_dir}
python ../gaudi_spawn.py \
    --world_size 8 --use_deepspeed run_clm_llama.py \
    --model_name_or_path /mnt/decapoda-research/llama-7b-hf \
    --cache_dir /mnt/hf_cache/ \
    --torch_dtype bfloat16 \
    --use_cache False \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --block_size 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --learning_rate 5e-5 \
    --output_dir ${res_dir} \
    --overwrite_output_dir \
    --gaudi_config_name Habana/gpt2 \
    --use_habana \
    --use_lazy_mode True \
    --use_hpu_graphs_for_training False\
    --adjust_throughput True \
    --pipelining_fwd_bwd True \
    --non_blocking_data_copy False \
    --throughput_warmup_steps 3 \
    --save_strategy  no \
    --logging_steps 10 \
    --max_steps 200 \
    --deepspeed ds_config.json \
    2>&1 \
    | tee ${res_dir}/run.log

res_dir=test-llama-lazy-pipe-blocking
mkdir -p ${res_dir}
python ../gaudi_spawn.py \
    --world_size 8 --use_deepspeed run_clm_llama.py \
    --model_name_or_path /mnt/decapoda-research/llama-7b-hf \
    --cache_dir /mnt/hf_cache/ \
    --torch_dtype bfloat16 \
    --use_cache False \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --block_size 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --learning_rate 5e-5 \
    --output_dir ${res_dir} \
    --overwrite_output_dir \
    --gaudi_config_name Habana/gpt2 \
    --use_habana \
    --use_lazy_mode True \
    --use_hpu_graphs_for_training False\
    --adjust_throughput True \
    --pipelining_fwd_bwd True \
    --non_blocking_data_copy True \
    --throughput_warmup_steps 3 \
    --save_strategy  no \
    --logging_steps 10 \
    --max_steps 200 \
    --deepspeed ds_config.json \
    2>&1 \
    | tee ${res_dir}/run.log

res_dir=test-llama-lazy-pipe-blocking-graph
mkdir -p ${res_dir}
python ../gaudi_spawn.py \
    --world_size 8 --use_deepspeed run_clm_llama.py \
    --model_name_or_path /mnt/decapoda-research/llama-7b-hf \
    --cache_dir /mnt/hf_cache/ \
    --torch_dtype bfloat16 \
    --use_cache False \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --block_size 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --learning_rate 5e-5 \
    --output_dir ${res_dir} \
    --overwrite_output_dir \
    --gaudi_config_name Habana/gpt2 \
    --use_habana \
    --use_lazy_mode True \
    --use_hpu_graphs_for_training True\
    --adjust_throughput True \
    --pipelining_fwd_bwd True \
    --non_blocking_data_copy True \
    --throughput_warmup_steps 3 \
    --save_strategy  no \
    --logging_steps 10 \
    --max_steps 200 \
    --deepspeed ds_config.json \
    2>&1 \
    | tee ${res_dir}/run.log

res_dir=test-llama-lazy-pipe-blocking-graph-acc
mkdir -p ${res_dir}
python ../gaudi_spawn.py \
    --world_size 8 --use_deepspeed run_clm_llama.py \
    --model_name_or_path /mnt/decapoda-research/llama-7b-hf \
    --cache_dir /mnt/hf_cache/ \
    --torch_dtype bfloat16 \
    --use_cache False \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --block_size 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --learning_rate 5e-5 \
    --output_dir ${res_dir} \
    --overwrite_output_dir \
    --gaudi_config_name Habana/gpt2 \
    --use_habana \
    --use_lazy_mode True \
    --use_hpu_graphs_for_training True\
    --adjust_throughput True \
    --pipelining_fwd_bwd True \
    --non_blocking_data_copy True \
    --throughput_warmup_steps 3 \
    --save_strategy  no \
    --logging_steps 10 \
    --max_steps 2000 \
    --deepspeed ds_config.json \
    2>&1 \
    | tee ${res_dir}/run.log
