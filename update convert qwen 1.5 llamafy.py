# coding=utf-8

# Converts the Qwen models to the format of the new generation model.
# Usage: python new_generation_convert.py --input_dir input --output_dir output

import json
import os
from collections import OrderedDict
from typing import Any, Dict, Optional

import fire
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers.modeling_utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    shard_checkpoint,
)
from transformers.utils import check_min_version

try:
    check_min_version("4.34.0")
except Exception:
    raise ValueError("Please upgrade transformers to 4.34.0")

CONFIG_NAME = "config.json"

def save_weight(input_dir: str, output_dir: str, shard_size: str, save_safetensors: bool) -> str:
    qwen_state_dict: Dict[str, torch.Tensor] = OrderedDict()
    for filepath in tqdm(os.listdir(input_dir), desc="Load weights"):
        if os.path.isfile(os.path.join(input_dir, filepath)) and filepath.endswith(".safetensors"):
            with safe_open(os.path.join(input_dir, filepath), framework="pt", device="cpu") as f:
                for key in f.keys():
                    qwen_state_dict[key] = f.get_tensor(key)

    new_model_state_dict: Dict[str, torch.Tensor] = OrderedDict()
    torch_dtype = None
    for key, value in tqdm(qwen_state_dict.items(), desc="Convert format"):
        if torch_dtype is None:
            torch_dtype = value.dtype
        # Adjust the layer names based on the new model's architecture
        # This is a simplified example; actual mappings may vary and need to be customized
        if "wte" in key:
            new_model_state_dict["new_model.embed_tokens.weight"] = value
        elif "ln_f" in key:
            new_model_state_dict["new_model.final_layer_norm.weight"] = value
        else:
            key = key.replace("transformer.h", "new_model.layers")
            if "attn.c_attn" in key:
                proj_size = value.size(0) // 3
                new_model_state_dict[key.replace("attn.c_attn", "self_attn.q_proj")] = value[:proj_size, ...]
                new_model_state_dict[key.replace("attn.c_attn", "self_attn.k_proj")] = value[proj_size : 2 * proj_size, ...]
                new_model_state_dict[key.replace("attn.c_attn", "self_attn.v_proj")] = value[2 * proj_size :, ...]
            elif "attn.c_proj" in key:
                new_model_state_dict[key.replace("attn.c_proj", "self_attn.o_proj")] = value
                # Assuming the new model requires bias for self_attn.o_proj, which is not present in Qwen
                new_model_state_dict[key.replace("attn.c_proj.weight", "self_attn.o_proj.bias")] = torch.zeros_like(value[:, 0]).squeeze()
            elif "ln_1" in key:
                new_model_state_dict[key.replace("ln_1", "input_layernorm")] = value
            elif "ln_2" in key:
                new_model_state_dict[key.replace("ln_2", "post_attention_layernorm")] = value
            elif "mlp.w1" in key:
                new_model_state_dict[key.replace("mlp.w1", "mlp.up_proj")] = value
            elif "mlp.w2" in key:
                new_model_state_dict[key.replace("mlp.w2", "mlp.gate_proj")] = value
            elif "mlp.c_proj" in key:
                new_model_state_dict[key.replace("mlp.c_proj", "mlp.down_proj")] = value
            elif "lm_head" in key:
                new_model_state_dict[key] = value
            else:
                raise KeyError("Unable to process key {}".format(key))

    weights_name = SAFE_WEIGHTS_NAME if save_safetensors else WEIGHTS_NAME
    shards, index = shard_checkpoint(new_model_state_dict, max_shard_size=shard_size, weights_name=weights_name)

    for shard_file, shard in tqdm(shards.items(), desc="Save weights"):
        if save_safetensors:
            save_file(shard, os.path.join(output_dir, shard_file), metadata={"format": "pt"})
        else:
            torch.save(shard, os.path.join(output_dir, shard_file))

    if index is None:
        print("Model weights saved in {}".format(os.path.join(output_dir, weights_name)))
    else:
        index_name = SAFE_WEIGHTS_INDEX_NAME if save_safetensors else WEIGHTS_INDEX_NAME
        with open(os.path.join(output_dir, index_name), "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, sort_keys=True)
        print("Model weights saved in {}".format(output_dir))

    return str(torch_dtype).replace("torch.", "")

def save_config(input_dir: str, output_dir: str, torch_dtype: str):
    with open(os.path.join(input_dir, CONFIG_NAME), "r", encoding="utf-8") as f:
        qwen_config_dict: Dict[str, Any] = json.load(f)

    new_model_config_dict: Dict[str, Any] = OrderedDict()
    # Adjust configuration parameters as necessary for the new model
    new_model_config_dict["architectures"] = ["NewModelForCausalLM"]
    new_model_config_dict["hidden_act"] = "silu"
    new_model_config_dict["hidden_size"] = qwen_config_dict["hidden_size"]
    new_model_config_dict["initializer_range"] = qwen_config_dict["initializer_range"]
    new_model_config_dict["intermediate_size"] = qwen_config_dict["intermediate_size"] // 2
    new_model_config_dict["max_position_embeddings"] = qwen_config_dict["max_position_embeddings"]
    new_model_config_dict["model_type"] = "new_model"
    new_model_config_dict["num_attention_heads"] = qwen_config_dict["num_attention_heads"]
    new_model_config_dict["num_hidden_layers"] = qwen_config_dict["num_hidden_layers"]
    new_model_config_dict["num_key_value_heads"] = qwen_config_dict["hidden_size"] // qwen_config_dict["kv_channels"]
    new_model_config_dict["pretraining_tp"] = 1
    new_model_config_dict["rms_norm_eps"] = qwen_config_dict["layer_norm_epsilon"]
    new_model_config_dict["rope_scaling"] = None
    new_model_config_dict["tie_word_embeddings"] = qwen_config_dict["tie_word_embeddings"]
    new_model_config_dict["torch_dtype"] = torch_dtype
    new_model_config_dict["transformers_version"] = "4.34.0"
    new_model_config_dict["use_cache"] = True
    new_model_config_dict["vocab_size"] = qwen_config_dict["vocab_size"]
    new_model_config_dict["attention_bias"] = True

    with open(os.path.join(output_dir, CONFIG_NAME), "w", encoding="utf-8") as f:
        json.dump(new_model_config_dict, f, indent=2)
    print("Model config saved in {}".format(os.path.join(output_dir, CONFIG_NAME)))

def llamafy_qwen(
    input_dir: str, 
    output_dir: str, 
    shard_size: Optional[str] = "2GB", 
    save_safetensors: Optional[bool] = False
):
    try:
        os.makedirs(output_dir, exist_ok=False)
    except Exception as e:
        raise Exception("Output dir already exists", e)

    torch_dtype = save_weight(input_dir, output_dir, shard_size, save_safetensors)
    save_config(input_dir, output_dir, torch_dtype)

if __name__ == "__main__":
    fire.Fire(llamafy_qwen)
