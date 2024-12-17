import os
import sys

import torch
import transformers
from transformers import GenerationMixin
import pyvene as pv
from pyvene.models.intervenable_modelcard import type_to_module_mapping, type_to_dimension_mapping
from pyvene.models.constants import CONST_INPUT_HOOK, CONST_OUTPUT_HOOK

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.modeling_chameleon import ChameleonForConditionalGeneration
from SALMONN.models.salmonn import SALMONN

chameleon_type_to_module_mapping = {
    "block_input": ("model.layers[%s]", CONST_INPUT_HOOK),
    "block_output": ("model.layers[%s]", CONST_OUTPUT_HOOK),
}

salmonn_type_to_module_mapping = {
    "block_input": ("llama_model.base_model.model.model.layers[%s]", CONST_INPUT_HOOK),
    "block_output": ("llama_model.base_model.model.model.layers[%s]", CONST_OUTPUT_HOOK),
}

chameleon_type_to_dimension_mapping = {
    "n_head": ("num_attention_heads",),
    "n_kv_head": ("num_key_value_heads",),
    "block_input": ("hidden_size",),
    "block_output": ("hidden_size",),
    "mlp_activation": ("intermediate_size",),
    "mlp_output": ("hidden_size",),
    "mlp_input": ("hidden_size",),
    "attention_value_output": ("hidden_size",),
    "head_attention_value_output": ("hidden_size/num_attention_heads",),
    "attention_output": ("hidden_size",),
    "attention_input": ("hidden_size",),
    "query_output": ("hidden_size",),
    "key_output": ("hidden_size",),
    "value_output": ("hidden_size",),
    "head_query_output": ("hidden_size/num_attention_heads",),
    "head_key_output": ("hidden_size/num_attention_heads",),
    "head_value_output": ("hidden_size/num_attention_heads",),
}

type_to_module_mapping[ChameleonForConditionalGeneration] = chameleon_type_to_module_mapping
type_to_dimension_mapping[ChameleonForConditionalGeneration] = chameleon_type_to_dimension_mapping
type_to_module_mapping[SALMONN] = salmonn_type_to_module_mapping
type_to_dimension_mapping[SALMONN] = type_to_dimension_mapping[
    transformers.models.llama.modeling_llama.LlamaForCausalLM
]


def intervened_generate_raw(
    prefix: torch.Tensor | dict,
    intervention: torch.Tensor,
    model: GenerationMixin | SALMONN,
    generation_config: dict,
    layers: int | list[int],
    position: int | list,
    intervention_type=pv.AdditionIntervention,
    forward_only=False,  # if True, only do one forward pass
):
    assert isinstance(position, (int, list))
    if isinstance(layers, int):
        layers = [layers]
    assert all(layer >= -1 for layer in layers)
    layer_configs = [
        (
            {"layer": layer, "component": "block_output"}
            if layer >= 0
            else {"layer": 0, "component": "block_input"}
        )
        for layer in layers
    ]
    pv_config = pv.IntervenableConfig(layer_configs, intervention_types=intervention_type)
    if isinstance(model, SALMONN):
        model.config = model.llama_model.config
    pv_model = pv.IntervenableModel(pv_config, model=model)

    position_idx = position
    if position_idx == -1:
        prefix_len = len(prefix["input_ids"][0])
        position_idx = prefix_len - 1
    if isinstance(position_idx, list):
        assert isinstance(position_idx[0], int)
        # [[[0, 1, 2, 3]]]
        position_idx = [[position_idx]] * len(layer_configs)

    if forward_only:
        intervened_outputs = pv_model(
            base=prefix,
            unit_locations={"base": position_idx},
            source_representations=intervention,
        )[1]
    else:
        intervened_outputs = pv_model.generate(
            base=prefix,
            unit_locations={"base": position_idx},
            source_representations=intervention,
            intervene_on_prompt=True,
            **generation_config,
        )[1]
    return intervened_outputs


def intervened_generate(
    tokenizer,
    model,
    prompt,
    pos_hidden,
    neg_hidden,
    strength,
    target_layer,
    position,
    generation_config,
    intervention_type=pv.AdditionIntervention,
):
    if isinstance(prompt, str):
        prompt = tokenizer(prompt, return_tensors="pt").to(model.device)

    intervened_output = intervened_generate_raw(
        prompt,
        pos_hidden * strength - neg_hidden * strength,
        model,
        generation_config,
        target_layer,
        position,
        intervention_type=intervention_type,
    )
    assert len(intervened_output) == 1
    if model.name_or_path == "meta-llama/Meta-Llama-3-8B":
        # there's a bug (?) with decoding directly:
        # >>> tokenizer.decode(tokenizer("a ' '")["input_ids"])
        # "<|begin_of_text|>a''"
        output = (
            "".join(tokenizer.convert_ids_to_tokens(intervened_output[0]))
            .replace("Ġ", " ")
            .replace("Ċ", "\n")
            .replace("âŀŀ", "➞")
            .replace("âī¤", "≤")
            .replace("č", "\r")
            .replace("ĉ", "\t")
            .replace("<|begin_of_text|>", "")
        )
    else:
        output = tokenizer.decode(intervened_output[0], skip_special_tokens=True)
    return output
