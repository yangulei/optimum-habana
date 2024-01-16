import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers import AutoConfig
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.gpu_migration
import time
import os
import argparse

# Tweak generation so that it runs faster on Gaudi
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
#from habana_frameworks.torch.hpu.metrics import metrics_dump
from habana_frameworks.torch.hpu.metrics import metric_global
from habana_frameworks.torch.utils.experimental import detect_recompilation_auto_model

from optimum.habana.checkpoint_utils import (
    get_ds_injection_policy,
    get_repo_root,
    model_is_optimized,
    model_on_meta,
    write_checkpoints_json,
)

def override_print(enable):
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if force or enable:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model (on the HF Hub or locally).",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        type=str,
        help='Argument to give a prompt of your choice as input(eg: --prompt "Hello world")',
    )
    args = parser.parse_args()
    
    # If the DeepSpeed launcher is used, the env variable _ will be equal to /usr/local/bin/deepspeed
    # For multi node, the value of the env variable WORLD_SIZE should be larger than 8
    use_deepspeed = "deepspeed" in os.environ["_"] or (
        "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 8
    )

    # Get world size, rank and local rank
    from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

    world_size, rank, local_rank = initialize_distributed_hpu()
    override_print(rank==0)

    if use_deepspeed:
        # Check if DeepSpeed is installed
        from transformers.integrations.deepspeed import is_deepspeed_available

        if not is_deepspeed_available():
            raise ImportError(
                "This script requires deepspeed: `pip install"
                " git+https://github.com/HabanaAI/DeepSpeed.git@1.13.0`."
            )
        import deepspeed

        # Initialize process(es) for DeepSpeed
        deepspeed.init_distributed(dist_backend="hccl")
        print("DeepSpeed is enabled.")
    else:
        print("Single-device run.")

    adapt_transformers_to_gaudi()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    model_kwargs = {
        "revision": "main",
        "token": None,
    }

    #if use_deepspeed:
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True, **model_kwargs)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="cpu", torch_dtype=torch.bfloat16, trust_remote_code=True, **model_kwargs)
    model = model.eval()

    # Initialize the model
    ds_inference_kwargs = {"dtype": torch.bfloat16}
    ds_inference_kwargs["tensor_parallel"] = {"tp_size": world_size}
    ds_inference_kwargs["enable_cuda_graph"] = True

    # Make sure all devices/nodes have access to the model checkpoints
    torch.distributed.barrier()

    ds_inference_kwargs["injection_policy"] = {}

    model = deepspeed.init_inference(model, **ds_inference_kwargs)
    model = model.module

    model.generation_config = GenerationConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model.generation_config.do_sample = False
    model.generation_config.static_shapes = 1
    model.generation_config.bucket_size = 512
    model.generation_config.ignore_eos = False
    model.generation_config.reuse_cache = False
    model.generation_config.trim_logits = False
    model.generation_config.attn_softmax_bf16 = False
    model.generation_config.limit_hpu_graphs = False
    model.generation_config.use_cache = True
    model.generation_config.use_flash_attention = False

    for i in range(2):
        prompt = args.prompt
        
        start=time.time()

        response, history = model.chat(tokenizer, prompt, history=None, generation_config=model.generation_config)
        end=time.time()
        if i == 1:
            token_count= len(tokenizer(response).input_ids)
            throughput = token_count / (end-start)
            print(response)
            print(f"Generated {token_count} tokens takes {end-start}, {throughput} tokens/s")
