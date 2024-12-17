import argparse
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, WhisperFeatureExtractor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.modeling_llava import LlavaForConditionalGeneration
from lib.modeling_chameleon import ChameleonForConditionalGeneration
from lib.utils import min_value_of_dtype, masked_log_softmax
from SALMONN.config import Config
from SALMONN.models.salmonn import SALMONN


def init_model(path, max_seq_len=None):
    token = os.getenv("HF_TOKEN")
    # The slightly different settings of the kwargs are for historical reasons and likely don't
    # matter much.
    if "llava" in path.lower():
        tokenizer = AutoProcessor.from_pretrained(path)
        model = LlavaForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto"
        )
    elif "chameleon" in path.lower():
        tokenizer = AutoProcessor.from_pretrained(path)
        model = ChameleonForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto"
        )
    elif "salmonn" in path.lower():
        args = argparse.Namespace(cfg_path=path, device="cuda:0", options=[])
        cfg = Config(args)
        model = SALMONN.from_config(cfg.config.model).eval().cuda()
        tokenizer = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            path, device_map="auto", trust_remote_code=True, token=token
        ).eval()
        if max_seq_len is not None:
            tokenizer.model_max_length = max_seq_len
    return tokenizer, model


@torch.inference_mode()
def get_emb_and_hidden_states(text, tokenizer, model, add_special_tokens=True):
    inputs = tokenizer(text, add_special_tokens=add_special_tokens, return_tensors="pt").to(
        model.device
    )
    hidden_states = model(inputs.input_ids, output_hidden_states=True).hidden_states
    return torch.stack([h.squeeze(0) for h in hidden_states], dim=0)


@torch.inference_mode()
def get_hidden_logprobs(text, tokenizer, model, add_special_tokens=True):
    all_hidden_states = get_emb_and_hidden_states(
        text, tokenizer, model, add_special_tokens=add_special_tokens
    )[1:]
    unembeddings = model.get_output_embeddings().weight.detach()
    all_hidden_logprobs = []
    for hidden_states in all_hidden_states:
        dist = hidden_states @ unembeddings.T
        hidden_logprobs = dist.float().log_softmax(dim=-1)
        all_hidden_logprobs.append(hidden_logprobs)
    return torch.stack(all_hidden_logprobs, dim=0)


@torch.inference_mode()
def get_logprobs(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output = model(**inputs, return_dict=True)
    assert output.logits.shape[0] == 1
    return output.logits[0].log_softmax(dim=-1)


@torch.inference_mode()
def get_text_image_logprobs(text, image, processor, model):
    inputs = processor(text, image, return_tensors="pt").to(device=model.device, dtype=model.dtype)
    inputs["labels"] = inputs["input_ids"]
    if isinstance(model, LlavaForConditionalGeneration):
        log_probs, ids = get_llava_hidden_logprobs(
            inputs, model
        )  # logprobs: (n_layers, n_tokens, vocab_size)
        image_tokens = (ids == 32000).squeeze(0)  # (n_tokens,)
    elif isinstance(model, ChameleonForConditionalGeneration):
        added_token_ids = list(processor.tokenizer.added_tokens_decoder.keys())
        img_token_ids = [
            k
            for k, v in processor.tokenizer.added_tokens_decoder.items()
            if v.content.startswith("IMGIMG")
        ]
        assert len(img_token_ids) == img_token_ids[-1] - img_token_ids[0] + 1
        log_probs, ids = get_chameleon_hidden_logprobs(
            inputs, model, masked_out_tokens=added_token_ids
        )
        image_tokens = ((ids >= img_token_ids[0]) & (ids <= img_token_ids[-1])).squeeze(0)
    else:
        assert False
    image_logprobs = log_probs[:, image_tokens]  # (n_layers, n_image_tokens, vocab_size)
    return image_logprobs


@torch.inference_mode()
def get_image_logprobs(image, processor, model):
    if isinstance(model, LlavaForConditionalGeneration):
        prompt = "USER: What is in the image?\n<image> ASSISTANT:"
    elif isinstance(model, ChameleonForConditionalGeneration):
        prompt = "What is in the image?\n<image>"
    else:
        assert False
    return get_text_image_logprobs(prompt, image, processor, model)


def first_token_id_for_string(tokenizer, s, add_prefix_space=True):
    if "llama-2" in tokenizer.name_or_path.lower():
        if add_prefix_space:
            return tokenizer(s, add_special_tokens=False)["input_ids"][0]
        else:
            space_asterisk_id = tokenizer("*", add_special_tokens=False)["input_ids"][0]
            tokens = tokenizer("*" + s, add_special_tokens=False)["input_ids"]
            assert tokens[0] == space_asterisk_id
            return tokens[1]
    elif "llama-3" in tokenizer.name_or_path.lower():
        return tokenizer((" " if add_prefix_space else "") + s, add_special_tokens=False)[
            "input_ids"
        ][0]
    else:
        assert False


def get_unembedding(token, model, tokenizer):
    tokenized = tokenizer(token, add_special_tokens=False, return_tensors="pt")
    input_ids = tokenized["input_ids"][0]
    if isinstance(model, SALMONN):
        model = model.llama_model
    return model.get_output_embeddings().weight[input_ids[0]]


@torch.inference_mode()
def get_llava_hidden_logprobs(inputs, model):
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    all_hidden_states = torch.stack([h.squeeze(0) for h in hidden_states], dim=0)[1:]

    unembeddings = model.get_output_embeddings().weight.detach()
    all_hidden_logprobs = []
    for hidden_states in all_hidden_states:
        # make sure they are in the same device.
        unembeddings = unembeddings.to(hidden_states)
        dist = hidden_states @ unembeddings.T
        hidden_logprobs = dist.float().log_softmax(dim=-1)
        all_hidden_logprobs.append(hidden_logprobs)

    ids = outputs.input_ids_cache
    return torch.stack(all_hidden_logprobs, dim=0), ids


@torch.inference_mode()
def get_chameleon_hidden_logprobs(inputs, model, masked_out_tokens: list[int] = None):
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    all_hidden_states = torch.stack([h.squeeze(0) for h in hidden_states], dim=0)[1:]

    unembeddings = model.get_output_embeddings().weight.detach()
    all_hidden_logprobs = []
    for hidden_states in all_hidden_states:
        # make sure they are in the same device.
        unembeddings = unembeddings.to(hidden_states)
        dist = hidden_states @ unembeddings.T
        if masked_out_tokens is not None:
            mask = torch.ones(dist.shape[-1], dtype=torch.bool, device=dist.device)
            mask[masked_out_tokens] = False
            hidden_logprobs = masked_log_softmax(dist.float(), mask=mask.unsqueeze(0), dim=-1)
            hidden_logprobs.masked_fill_(
                ~mask.unsqueeze(0), min_value_of_dtype(hidden_logprobs.dtype)
            )
        else:
            hidden_logprobs = dist.float().log_softmax(dim=-1)
        all_hidden_logprobs.append(hidden_logprobs)

    ids_out = outputs.ids_out
    return torch.stack(all_hidden_logprobs, dim=0), ids_out
