#!/bin/bash

Help()
{
    # Display Help
    echo "Get the inference check table data for a huggingface model based on optimum-habana."
    echo
    echo "Syntax: get_check_table_data.sh <-n|p|s|m> [-l]"
    echo "options:"
    echo "n     Name of the model."
    echo "p     Path of the model."
    echo "s     Script for the generation."
    echo "m     Mode for benchmarking (latency|throughput)."
    echo "l     With --limit_hpu_graphs or not."
    echo
}

# define variables
model_name=""
model_path=""
run_generation=""
benchmark_mode=""
limit_hpu_graphs=false

# Get the flags
while getopts h:n:p:s:m:l flag; do
    case $flag in
        h) # display Help
            Help
            exit;;
        n) # get model name
            model_name=$OPTARG;;
        p) # get model path
            model_path=$OPTARG;;
        s) # get generation script
            run_generation=$OPTARG;;
        m) # get benchmark mode
            benchmark_mode=$OPTARG;;
        l) # get benchmark mode
            limit_hpu_graphs=true;;
        \?) # Invalid option
            echo "Error: Invalid option"
            Help
            exit;;
    esac
done

# check the flags and set related variables
if [[ $model_name = "" ]]; then
    echo empty model name.
    exit -1
fi

if [[ ! -d $model_path ]]; then
    echo invalid model path.
    exit -1
fi

if [[ ! -r $run_generation ]]; then
    echo invalid generation script.
    exit -1
fi

if [[ $benchmark_mode = "latency" ]]; then
    in_out_sizes=("32,1" "64,1" "128,1" "256,1" "512,1" "1024,1" "2048,1")
elif [[ $benchmark_mode = "throughput" ]]; then
    in_out_sizes=("32,68" "64,64" "128,128" "256,256" "512,512" "1024,1024" "2048,2048")
else
    echo "invalid benchmark mode, chose 'latency' or 'throughput'."
    exit -1
fi

arg_limit_hpu_graphs=""
if $limit_hpu_graphs; then
    arg_limit_hpu_graphs="--limit_hpu_graphs"
    echo "Benchmarking ${benchmark_mode} for ${model_name} from ${model_path} using ${run_generation} with --limit_hpu_graphs."
    res_root=res-gaudi2_${model_name}_inference_${benchmark_mode}_limit_hpu_graphs
else
    echo "Benchmarking ${benchmark_mode} for ${model_name} from ${model_path} using ${run_generation} without --limit_hpu_graphs."
    res_root=res-gaudi2_${model_name}_inference_${benchmark_mode}
fi

model_path=`realpath $model_path`
run_generation=`realpath $run_generation`

echo Saving results to ${res_root}
mkdir -p $res_root
cd $res_root

# Benchmark with specific BS,ISL,OSL combination
in_out_sizes=("32,68" "64,64" "128,128" "256,256" "512,512" "1024,1024" "2048,2048")
batch_sizes=(1 2 4 8 16 32 64 128 256 512)
for in_out_dims in ${in_out_sizes[@]}; do
    for bs in ${batch_sizes[@]}; do
        input_tokens=$(echo $in_out_dims | awk -F',' '{ print $1 }')
        output_tokens=$(echo $in_out_dims | awk -F',' '{ print $2 }')

        res_dir=gaudi2_${model_name}_I_bs${bs}_in${input_tokens}_out${output_tokens}_bw1_1c
        log=${res_dir}/run.log
        if [ -d "$res_dir" ]; then
            if [[ `grep ERROR $log` ]] || [[ `grep Throughput $log` ]]; then
                echo "results already exists, skip."
                continue
            fi
        fi
        mkdir -p ${res_dir}

        echo "Benchmarking with bs=${bs}, input_tokens=${input_tokens}, output_tokens=${output_tokens} with --limit_hpu_graphs"
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
            --attn_softmax_bf16 \
            ${arg_limit_hpu_graphs} \
            --reuse_cache \
            --trim_logits \
            --warmup 0 \
            --profiling_warmup_steps 0 \
            --profiling_steps 0 \
            --output_dir ${res_dir} \
            --prompt "Hello world" \
            2>&1 \
            | tee $log
        mv .graph_dumps hpu_profile checkpoints.json ${res_dir}

        if [[ `grep ERROR $log` ]]
        then
            echo ERROR found for benchmarking for bs=${bs}, input_tokens=${input_tokens}, output_tokens=${output_tokens}, skip larger BS
            break
        else
            echo Done benchmarking for bs=${bs}, input_tokens=${input_tokens}, output_tokens=${output_tokens}
        fi
    done    # bs
done    # in_out_sizes
