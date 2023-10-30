# set -e

for zero in {1,2,3}; do
    for mb in {1,2,4,8}; do
        for perf_warmup in {0,15}; do
            perf_steps=2
            res_dir=res-clm-gpt-neox-20b-zero${zero}-mb${mb}-ddp8-perf${perf_warmup}-$((${perf_warmup}+${perf_steps}))
            if [ -d "$res_dir" ]; then
                continue
            fi
            mkdir -p ${res_dir}

            GRAPH_VISUALIZATION=1 \
            PT_HPU_POOL_MEM_ACQUIRE_PERC=100 \
            python ../gaudi_spawn.py \
                --world_size 4 --use_deepspeed run_clm.py \
                --model_name_or_path EleutherAI/gpt-neox-20b \
                --cache_dir /hf_cache/ \
                --dataset_name wikitext \
                --dataset_config_name wikitext-2-raw-v1 \
                --block_size 1024 \
                --per_device_train_batch_size ${mb} \
                --do_train \
                --do_eval False \
                --output_dir ${res_dir} \
                --overwrite_output_dir \
                --gaudi_config_name Habana/gpt2 \
                --use_habana \
                --use_lazy_mode \
                --gradient_checkpointing \
                --throughput_warmup_steps 3 \
                --profiling_warmup_steps ${perf_warmup} \
                --profiling_steps ${perf_steps} \
                --profiling_record_shapes False \
                --save_strategy  no \
                --logging_steps 10 \
                --max_steps 20 \
                --deepspeed ds_config_zero_${zero}_offload.json \
                2>&1 \
                | tee ${res_dir}/run.log
            rm ${res_dir}/pytorch_model*
            if [ ${zero} -eq 3 ]; then
                rm -r ${res_dir}/global_step*
            fi
            if [ ${perf_warmup} -gt 0 ]; then
                cd hpu_profile
                for fn in *.json; do
                    habana_perf_tool --trace ${fn} 2>&1 | tee ../summary_${fn}.log
                done
                cd ..
                mv .graph_dumps hpu_profile* *.log ${res_dir}
            else
                mv .graph_dumps ${res_dir}
            fi
        done
    done
done


