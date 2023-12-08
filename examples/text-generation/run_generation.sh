#!/bin/bash

# Set variables
model_name="llama2-7b"
model_path="/models/llama2-7b"
bs=1
input_tokens=32
output_tokens=68
run_generation="run_generation.py"
limit_hpu_graphs=false

arg_limit_hpu_graphs=""
if $limit_hpu_graphs; then
    arg_limit_hpu_graphs="--limit_hpu_graphs"
    echo "Benchmarking with bs=${bs}, input_tokens=${input_tokens} and output_tokens=${output_tokens} for ${model_name} from ${model_path} using ${run_generation} with --limit_hpu_graphs."
    res_dir=res-gaudi2_${model_name}_inference_bs${bs}_in${input_tokens}_out${output_tokens}_bw1_1c_limit_hpu_graphs
else
    echo "Benchmarking bs=${bs}, input_tokens=${input_tokens} and output_tokens=${output_tokens} for ${model_name} from ${model_path} using ${run_generation} without --limit_hpu_graphs."
    res_dir=res-gaudi2_${model_name}_inference_bs${bs}_in${input_tokens}_out${output_tokens}_bw1_1c
fi

model_path=`realpath $model_path`
run_generation=`realpath $run_generation`
echo Full model path: $model_path
echo Full script path: $run_generation
echo arg_limit_hpu_graphs: $arg_limit_hpu_graphs
echo saving results to $res_dir

log=${res_dir}/run.log
if [ -d "$res_dir" ]; then
    if [[ `grep "Allocation failed" $log` ]] || [[ `grep Throughput $log` ]]; then
        echo "results already exists, skip."
        exit -1
    fi
fi
mkdir -p ${res_dir}

HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_EVALUATE_OFFLINE=1 \
GRAPH_VISUALIZATION=1 \
PT_HPU_POOL_MEM_ACQUIRE_PERC=100 \
python ${run_generation} \
    --model_name_or_path ${model_path} \
    --max_input_tokens ${input_tokens} \
    --max_new_tokens ${output_tokens} \
    --bf16 \
    --use_hpu_graphs \
    --use_kv_cache \
    --batch_size ${bs} \
    --n_iterations 5 \
    ${arg_limit_hpu_graphs} \
    --reuse_cache \
    --trim_logits \
    --warmup 3 \
    --warmup_timestamps \
    --profiling_warmup_steps 0 \
    --profiling_steps 0 \
    --output_dir $res_dir \
    --prompt "Hello world" \
    > $log 2>&1
mv .graph_dumps hpu_profile checkpoints.json timestamps* $res_dir
