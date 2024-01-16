#!/bin/bash

Help()
{
    # Display Help
    echo "Run inference for a huggingface model based on optimum-habana."
    echo
    echo "Syntax: bash run_inference.sh <-m|p> [-n|d|b|i|o|l] [-h]"
    echo "options:"
    echo "m  Model name to inference"
    echo "p  Path of the model parameters"
    echo "n  Number of HPU to use, [1-8], default=1"
    echo "d  Data type, [fp8|bf16], default=bf16"
    echo "b  Batch size, int, default=1"
    echo "i  Input tokens, int, default=32"
    echo "o  Output tokens, int, default=32"
    echo "l  Enable limit_hpu_graphs option"
    echo "h  Help info"
    echo
}

# Set variables
model_name=""
model_path=""
num_hpu="1"
data_type="bf16"
batch_size="1"
input_tokens="32"
output_tokens="32"

limit_hpu_graphs=false
# Get the options
while getopts h:m:p:n:d:b:i:o:l flag; do
    case $flag in
        h) # display Help
            Help
            exit;;
        m) # get model name
            model_name=$OPTARG;;
        p) # get model path
            model_path=$OPTARG;;
        n) # get number of HPUs
            num_hpu=$OPTARG;;
        d) # get data type
            data_type=$OPTARG;;
        b) # get batch size
            batch_size=$OPTARG;;
        i) # get input length
            input_tokens=$OPTARG;;
        o) # get output length
            output_tokens=$OPTARG;;
        l) # limit hpu graph
            limit_hpu_graphs=true;;
        \?) # Invalid option
            echo "Error: Invalid option"
            Help
            exit;;
    esac
done

if [ $num_hpu -gt 1 ]; then
    export PT_HPU_LAZY_ACC_PAR_MODE=0
    export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
    spawn="../gaudi_spawn.py --use_deepspeed --world_size ${num_hpu}"
fi

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export PT_HPU_POOL_MEM_ACQUIRE_PERC=100
config="--use_hpu_graphs --use_kv_cache --attn_softmax_bf16 --reuse_cache --trim_logits --skip_hash_with_views"

if $limit_hpu_graphs; then
    config="$config --limit_hpu_graphs"
fi

if [ $data_type = "fp8" ]; then
    export USE_DEFAULT_QUANT_PARAM=true
    export UPDATE_GRAPH_OUTPUT_MME=false
    export ENABLE_CALC_DYNAMIC_RANGE=false
    export ENABLE_EXPERIMENTAL_FLAGS=true
    config="${config} --kv_cache_fp8"
fi

echo "Benchmarking ${model_name} with bs=${batch_size}, input_tokens=${input_tokens}, output_tokens=${output_tokens}, dtype=${data_type} using ${num_hpu} HPUs"

res_dir=./run_inference_logs/gaudi2_${model_name}_${data_type}_bs${batch_size}_in${input_tokens}_out${output_tokens}_${num_hpu}c

echo "Saving results to ${res_dir}"
mkdir -p ${res_dir}

python ${spawn} run_generation.py \
    --model_name_or_path ${model_path} \
    --${data_type} \
    --batch_size ${batch_size} \
    --max_input_tokens ${input_tokens} \
    --max_new_tokens ${output_tokens} \
    ${config} \
    --prompt "Hello world" \
    --output_dir ${res_dir} \
    2>&1 | tee ${res_dir}/run.log

