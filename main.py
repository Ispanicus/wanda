import argparse
import csv
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, modeling_utils
from transformers.utils.hub import convert_file_size_to_int
from safetensors.torch import storage_ptr, storage_size
from typing import Dict, Union, Tuple
from importlib.metadata import version

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers
from lib.eval import eval_belebele

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('bitsandbytes', version('bitsandbytes'))
print('# of gpus: ', torch.cuda.device_count())

def id_tensor_storage(tensor: torch.Tensor) -> Tuple[torch.device, int, int]:

    unique_id = storage_ptr(tensor)

    return tensor.device, unique_id, storage_size(tensor)

# Monkey patching the checkpoint function to shard the model weights
def custom_shard_checkpoint(
    state_dict: Dict[str, torch.Tensor], max_shard_size: Union[int, str] = "10GB", weights_name: str = "pytorch_model.bin"
):

    max_shard_size = convert_file_size_to_int(max_shard_size)

    sharded_state_dicts = [{}]
    last_block_size = 0
    total_size = 0
    storage_id_to_block = {}

    for key, weight in state_dict.items():
        # when bnb serialization is used the weights in the state dict can be strings
        # check: https://github.com/huggingface/transformers/pull/24416 for more details
        if isinstance(weight, str):
            continue
        else:
            storage_id = id_tensor_storage(weight)

        # If a `weight` shares the same underlying storage as another tensor, we put `weight` in the same `block`
        if storage_id in storage_id_to_block:
            block_id = storage_id_to_block[storage_id]
            sharded_state_dicts[block_id][key] = weight
            continue

        weight_size = weight.numel() * modeling_utils.dtype_byte_size(weight.dtype)

        # If this weight is going to tip up over the maximal size, we split, but only if we have put at least one
        # weight in the current shard.
        if last_block_size + weight_size > max_shard_size and len(sharded_state_dicts[-1]) > 0:
            sharded_state_dicts.append({})
            last_block_size = 0

        sharded_state_dicts[-1][key] = weight
        last_block_size += weight_size
        total_size += weight_size
        storage_id_to_block[storage_id] = len(sharded_state_dicts) - 1

    # If we only have one shard, we return it
    if len(sharded_state_dicts) == 1:
        return {weights_name: sharded_state_dicts[0]}, None

    # Otherwise, let's build the index
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):
        shard_file = weights_name.replace(".bin", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.bin")
        shard_file = shard_file.replace(
            ".safetensors", f"-{idx + 1:05d}-of-{len(sharded_state_dicts):05d}.safetensors"
        )
        shards[shard_file] = shard
        for key in shard.keys():
            weight_map[key] = shard_file

    # Add the metadata
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}
    return shards, index

def get_llm(model_name, cache_dir="llm_weights", quantize=False):

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto",
        load_in_8bit=quantize,
    )

    if "bloom" in model_name:
        model.seqlen = 2048
    else:
        model.seqlen = model.config.max_position_embeddings 
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", "ablate_magnitude", "ablate_wanda"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--quantize', type=int, default=0, help='Desired bit quantization')
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    if args.quantize:
        model = get_llm(args.model, args.cache_dir, quantize=True)
    else:
        model = get_llm(args.model, args.cache_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    # orig_ppl_train, orig_ppl_test = eval_ppl(model, tokenizer, device)

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    #ppl_train, ppl_test = eval_ppl(model, tokenizer, device)
    #print(f"original ppl on wikipedia_train {orig_ppl_train}, wikipedia_test {orig_ppl_test}")
    #print(f"ppl on wikipedia_train {ppl_train}, wikipedia_test {ppl_test}")

    if args.save:
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")

        with open(save_filepath, "w") as f:
            if "ablate" in args.prune_method:
                print("method\tactual_sparsity\tppl_train\tppl_test", file=f, flush=True)
                print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_train:.4f}\t{ppl_test:.4f}", file=f, flush=True)
            else:
                print("method\tactual_sparsity\tppl_test", file=f, flush=True)
                print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

    if args.save_model:
        if args.quantize:
            modeling_utils.shard_checkpoint = custom_shard_checkpoint
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
    
    model_path = model.config.name_or_path

    # Chooses batch size for different model sizes
    batch_size = next((size for name, size in {
        'bloom-560m': 4,
        'bloom-1b7': 4,
        'bloom-3b': 2,
        'bloom-7b1': 1
    }.items() if name in model_path), 1)

    answers = eval_belebele(model, tokenizer, BATCH_SIZE=batch_size, quantized=bool(args.quantize))

    with open(f'{args.save_model}/belebele.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'Answer', 'Label'])
        for key, value in answers.items():
            writer.writerow([key] + list(value))

if __name__ == '__main__':
    main()