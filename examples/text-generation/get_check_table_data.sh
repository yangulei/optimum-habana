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

# Set variables
model_name=""
model_path=""
run_generation=""
benchmark_mode=""
limit_hpu_graphs=false

# Get the options
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
echo Full model path: $model_path
echo Full script path: $run_generation
echo saving results to ${res_root}