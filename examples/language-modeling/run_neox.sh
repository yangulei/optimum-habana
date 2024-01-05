DATA_ROOT=/host/mnt/disk7
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_EVALUATE_OFFLINE=1 \
python ../gaudi_spawn.py \
    --hostfile hostfile --use_deepspeed run_clm.py \
    --model_name_or_path ${DATA_ROOT}/HF_models/gpt-neox-20b \
    --dataset_name ${DATA_ROOT}/saved_hf_datasets/small-the_pile-dedup/ \
    --dataset_config_name wikitext-2-raw-v1\
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --output_dir ${DATA_ROOT}/youlei/train_neox_bs1 \
    --gaudi_config_name ${DATA_ROOT}/HF_models/habana-gpt2 \
    --use_habana \
    --use_lazy_mode \
    --gradient_checkpointing \
    --use_hpu_graphs_for_inference \
    --throughput_warmup_steps 3 \
    --deepspeed ../../tests/configs/deepspeed_zero_2.json \
    2>&1 | tee train.log