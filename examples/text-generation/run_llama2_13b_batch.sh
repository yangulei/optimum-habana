
# single card inference
input_tokens=32
output_tokens=68
for bs in {1,2,4,8,16,32,64,128,256}; do
    # with --limit_hpu_graphs
    res_dir=gaudi2_llama2-13b_I_bs${bs}_in${input_tokens}_out${output_tokens}_bw1_1c_limit-graph_timestamp
    if [ -d "$res_dir" ]; then
        echo "results folder already exists."
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
        --limit_hpu_graphs \
        --reuse_cache \
        --trim_logits \
        --warmup 5 \
        --warmup_timestamp \
        --profiling_warmup_steps 0 \
        --profiling_steps 0 \
        --output_dir ${res_dir} \
        --prompt "Hello world" \
        2>&1 \
        | tee ${res_dir}/run.log
    mv .graph_dumps hpu_profile checkpoints.json timestamps* ${res_dir}

    # without --limit_hpu_graphs
    res_dir=gaudi2_llama2-13b_I_bs${bs}_in${input_tokens}_out${output_tokens}_bw1_1c_timestamp
    if [ -d "$res_dir" ]; then
        echo "results folder already exists."
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
        --warmup 5 \
        --warmup_timestamp \
        --profiling_warmup_steps 0 \
        --profiling_steps 0 \
        --output_dir ${res_dir} \
        --prompt "Hello world" \
        2>&1 \
        | tee ${res_dir}/run.log
    mv .graph_dumps hpu_profile checkpoints.json timestamps* ${res_dir}    
done

# deepspeed tp
input_tokens=32
output_tokens=68
for bs in {1,2,4,8,16,32,64,128,256}; do
    # with --limit_hpu_graphs
    res_dir=gaudi2_llama2-13b_I_bs${bs}_in${input_tokens}_out${output_tokens}_bw1_8c_limit-graph_timestamp
    if [ -d "$res_dir" ]; then
        echo "results folder already exists."
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
        --n_iterations 10 \
        --attn_softmax_bf16 \
        --limit_hpu_graphs \
        --reuse_cache \
        --trim_logits \
        --warmup 5 \
        --warmup_timestamp \
        --profiling_warmup_steps 0 \
        --profiling_steps 0 \
        --output_dir ${res_dir} \
        --prompt "Hello world" \
        2>&1 \
        | tee ${res_dir}/run.log
    mv .graph_dumps hpu_profile checkpoints.json timestamps* ${res_dir}

    # without --limit_hpu_graphs
    res_dir=gaudi2_llama2-13b_I_bs${bs}_in${input_tokens}_out${output_tokens}_bw1_8c_timestamp
    if [ -d "$res_dir" ]; then
        echo "results folder already exists."
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
        --n_iterations 10 \
        --attn_softmax_bf16 \
        --reuse_cache \
        --trim_logits \
        --warmup 5 \
        --warmup_timestamp \
        --profiling_warmup_steps 0 \
        --profiling_steps 0 \
        --output_dir ${res_dir} \
        --prompt "Hello world" \
        2>&1 \
        | tee ${res_dir}/run.log
    mv .graph_dumps hpu_profile checkpoints.json timestamps* ${res_dir}
done