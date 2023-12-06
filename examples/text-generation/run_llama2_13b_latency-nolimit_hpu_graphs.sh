
# deepspeed tp
# last_code=0
# last_bs=0
# for input_tokens in {1024,2048,3072,4096,8192}; do
#     for output_tokens in {1,512,1024,2048,4096,8192}; do
#         for bs in {1,4,16,64,256,512}; do
# for input_tokens in {1024,2048,3072,4096,8192}; do
    input_tokens=3072
    # for output_tokens in {1,512,1024,2048,4096,8192}; do
        output_tokens=1
        for bs in {1,2,4,8,16,32,64,128}; do
            res_dir=res-llama2-13b-tp-in${input_tokens}-out${output_tokens}-bs${bs}
            if [ -d "$res_dir" ]; then
                continue
            fi
            mkdir -p ${res_dir}
            
            HF_DATASETS_OFFLINE=1 \
            TRANSFORMERS_OFFLINE=1 \
            HF_EVALUATE_OFFLINE=1 \
            GRAPH_VISUALIZATION=1 \
            PT_HPU_POOL_MEM_ACQUIRE_PERC=100 \
            python ../gaudi_spawn.py --use_deepspeed --world_size 8 run_generation.py \
            --model_name_or_path /models/llama-2-13b-hf-bf16-sharded \
            --max_input_tokens ${input_tokens} \
            --max_new_tokens ${output_tokens} \
            --bf16 \
            --use_hpu_graphs \
            --use_kv_cache \
            --batch_size ${bs} \
            --n_iterations 20 \
            --attn_softmax_bf16 \
            --reuse_cache \
            --trim_logits \
            --profiling_warmup_steps 5 \
            --profiling_steps 5 \
            --output_dir ${res_dir} \
            --prompt "Hello world" \
            2>&1 \
            | tee ${res_dir}/run.log
            # last_code=$?
            # last_bs=${bs}
            # if [ ${last_code} -ne 0 ]; then
            #     rm -r .graph_dumps hpu_profile
            #     break
            # else
                mv .graph_dumps hpu_profile checkpoints.json ${res_dir}
            # fi
        done
    #     if [[${last_code} -ne 0 && ${last_bs} -eq 1 ]]; then
    #         break
    #     fi
    # done
# done

# single card inference
# last_code=0
# last_bs=0
# for input_tokens in {1024,2048,3076,4096,8192}; do
    input_tokens=3072
    # for output_tokens in {1,512,1024,2048,4096,8192}; do
        output_tokens=1
        # for bs in {1,4,16,32,64}; do
        for bs in {1,2,4,8,16}; do
            res_dir=res-llama2-13b-1c-in${input_tokens}-out${output_tokens}-bs${bs}
            if [ -d "$res_dir" ]; then
                continue
            fi
            mkdir -p ${res_dir}
            
            HF_DATASETS_OFFLINE=1 \
            TRANSFORMERS_OFFLINE=1 \
            HF_EVALUATE_OFFLINE=1 \
            GRAPH_VISUALIZATION=1 \
            PT_HPU_POOL_MEM_ACQUIRE_PERC=100 \
            python run_generation.py \
            --model_name_or_path /models/llama-2-13b-hf-bf16-sharded \
            --max_input_tokens ${input_tokens} \
            --max_new_tokens ${output_tokens} \
            --bf16 \
            --use_hpu_graphs \
            --use_kv_cache \
            --batch_size ${bs} \
            --n_iterations 10 \
            --attn_softmax_bf16 \
            --reuse_cache \
            --trim_logits \
            --profiling_warmup_steps 5 \
            --profiling_steps 5 \
            --output_dir ${res_dir} \
            --prompt "Hello world" \
            2>&1 \
            | tee ${res_dir}/run.log
            # last_code=$?
            # last_bs=${bs}
            # if [ ${last_code} -ne 0 ]; then
            #     rm -r .graph_dumps hpu_profile
            #     break
            # else
                mv .graph_dumps hpu_profile checkpoints.json ${res_dir}
            # fi
        done
    #     if [[${last_code} -ne 0 && ${last_bs} -eq 1 ]]; then
    #         break
    #     fi
    # done
# done


