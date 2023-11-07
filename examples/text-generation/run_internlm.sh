# python ../gaudi_spawn.py --use_deepspeed --world_size 8 run_generation.py \
# python run_generation.py \

# GRAPH_VISUALIZATION=1 \
# PT_HPU_POOL_MEM_ACQUIRE_PERC=100 \
# python ../gaudi_spawn.py --use_deepspeed --world_size 8 run_generation.py \
# --model_name_or_path /models/internlm-20b \
# --max_new_tokens 100 \
# --bf16 \
# --use_hpu_graphs \
# --use_kv_cache \
# --batch_size 4096 \
# --attn_softmax_bf16 \
# --limit_hpu_graphs \
# --reuse_cache \
# --trim_logits \

# deepspeed tp
for tokens in {64,128,256,512,1024,2048,4096}; do
    for bs in {1,2,4,8,16,32,64,128,256,512,1024,2048,4096}; do
        res_dir=res-internlm-20b-tp-tok${tokens}-bs${bs}
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
        --model_name_or_path /models/internlm-20b \
        --max_new_tokens ${tokens} \
        --bf16 \
        --use_hpu_graphs \
        --use_kv_cache \
        --batch_size ${bs} \
        --attn_softmax_bf16 \
        --limit_hpu_graphs \
        --reuse_cache \
        --trim_logits \
        --prompt "Hello world" \
        2>&1 \
        | tee ${res_dir}/run.log
        mv .graph_dumps ${res_dir}
    done
done


# native dp
for tokens in {64,128,256,512,1024,2048,4096}; do
    for bs in {1,2,4,8,16,32,64,128,256,512,1024,2048,4096}; do
        res_dir=res-internlm-20b-dp-tok${tokens}-bs${bs}
        if [ -d "$res_dir" ]; then
            continue
        fi
        mkdir -p ${res_dir}

        HF_DATASETS_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        HF_EVALUATE_OFFLINE=1 \
        GRAPH_VISUALIZATION=1 \
        PT_HPU_POOL_MEM_ACQUIRE_PERC=100 \
        python ../gaudi_spawn.py --world_size 8 run_generation.py \
        --model_name_or_path /models/internlm-20b \
        --max_new_tokens ${tokens} \
        --bf16 \
        --use_hpu_graphs \
        --use_kv_cache \
        --batch_size ${bs} \
        --attn_softmax_bf16 \
        --limit_hpu_graphs \
        --reuse_cache \
        --trim_logits \
        --prompt "Hello world" \
        2>&1 \
        | tee ${res_dir}/run.log
        mv .graph_dumps ${res_dir}
    done
done


# single card inference
for tokens in {64,128,256,512,1024,2048,4096}; do
    for bs in {1,2,4,8,16,32,64,128,256,512,1024,2048,4096}; do
        res_dir=res-internlm-20b-1c-tok${tokens}-bs${bs}
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
        --model_name_or_path /models/internlm-20b \
        --max_new_tokens ${tokens} \
        --bf16 \
        --use_hpu_graphs \
        --use_kv_cache \
        --batch_size ${bs} \
        --attn_softmax_bf16 \
        --limit_hpu_graphs \
        --reuse_cache \
        --trim_logits \
        --prompt "Hello world" \
        2>&1 \
        | tee ${res_dir}/run.log
        mv .graph_dumps ${res_dir}
    done
done


# deepspeed tp with profiling
for tokens in {64,256,1024,4096}; do
    for bs in {1,8,64,512,4096}; do
        res_dir=res-internlm-20b-tp-tok${tokens}-bs${bs}-prof
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
        --model_name_or_path /models/internlm-20b \
        --max_new_tokens ${tokens} \
        --bf16 \
        --use_hpu_graphs \
        --use_kv_cache \
        --batch_size ${bs} \
        --n_iterations 10 \
        --attn_softmax_bf16 \
        --limit_hpu_graphs \
        --reuse_cache \
        --trim_logits \
        --profiling_warmup_steps 5 \
        --profiling_steps 5 \
        --output_dir ${res_dir} \
        --prompt "Hello world" \
        2>&1 \
        | tee ${res_dir}/run.log
        mv .graph_dumps hpu_profile ${res_dir}
    done
done


# native dp with profiling
for tokens in {64,256,1024,4096}; do
    for bs in {1,8,64,512,4096}; do
        res_dir=res-internlm-20b-dp-tok${tokens}-bs${bs}-prof
        if [ -d "$res_dir" ]; then
            continue
        fi
        mkdir -p ${res_dir}
        
        HF_DATASETS_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        HF_EVALUATE_OFFLINE=1 \
        GRAPH_VISUALIZATION=1 \
        PT_HPU_POOL_MEM_ACQUIRE_PERC=100 \
        python ../gaudi_spawn.py --world_size 8 run_generation.py \
        --model_name_or_path /models/internlm-20b \
        --max_new_tokens ${tokens} \
        --bf16 \
        --use_hpu_graphs \
        --use_kv_cache \
        --batch_size ${bs} \
        --n_iterations 10 \
        --attn_softmax_bf16 \
        --limit_hpu_graphs \
        --reuse_cache \
        --trim_logits \
        --profiling_warmup_steps 5 \
        --profiling_steps 5 \
        --output_dir ${res_dir} \
        --prompt "Hello world" \
        2>&1 \
        | tee ${res_dir}/run.log
        mv .graph_dumps hpu_profile ${res_dir}
    done
done


# single card inference with profiling
for tokens in {64,256,1024,4096}; do
    for bs in {1,8,64,512}; do
        res_dir=res-internlm-20b-1c-tok${tokens}-bs${bs}-prof
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
        --model_name_or_path /models/internlm-20b \
        --max_new_tokens ${tokens} \
        --bf16 \
        --use_hpu_graphs \
        --use_kv_cache \
        --batch_size ${bs} \
        --n_iterations 10 \
        --attn_softmax_bf16 \
        --limit_hpu_graphs \
        --reuse_cache \
        --trim_logits \
        --profiling_warmup_steps 5 \
        --profiling_steps 5 \
        --output_dir ${res_dir} \
        --prompt "Hello world" \
        2>&1 \
        | tee ${res_dir}/run.log
        mv .graph_dumps hpu_profile ${res_dir}
    done
done