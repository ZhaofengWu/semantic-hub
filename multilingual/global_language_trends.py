import argparse
from collections import defaultdict
import numpy as np
import os
import pickle
import sys

from datasets import load_dataset
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multilingual.download_lm_corpus import LANGUAGES  # pylint: disable=wrong-import-position
from lib.logit_lens import init_model, get_hidden_logprobs


def load_wiki(lang, num_samples):
    data = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train")
    expanded_data = []
    idx = 0
    for text in tqdm(data):
        splitted = text["text"].split("\n")
        for split in splitted:
            if idx >= num_samples:
                break
            if len(split) > 0:
                expanded_data.append(split)
                idx += 1
    return expanded_data


def report_and_return(token_lang_probs, top_k=-1):
    entropy = -(token_lang_probs * torch.log2(token_lang_probs)).sum(dim=-1)
    # token_lang_probs: [n_layers, seq_len, n_langs]
    avg_token_lang_probs = token_lang_probs.sum(dim=1)  # divide later

    layer2entropies = dict()
    layer2toplangprob = defaultdict(dict)

    for i, probs in enumerate(avg_token_lang_probs):
        if top_k == -1:
            top_k = probs.shape[0]  # all languages
        top_langs = torch.topk(probs, top_k)
        layer2entropies[i] = entropy[i].sum().item()
        for lang_idx, prob in zip(top_langs.indices, top_langs.values):
            layer2toplangprob[i][LANGUAGES[lang_idx]] = prob
    return layer2entropies, layer2toplangprob


def calculate_log_prob(data, tokenizer, model, language_distributions_path):
    # 1/(\sum_sent len(sent)) \sum_{sent ∈ corpus} \sum_{token position ∈ sent} p(lang | token position)
    # = 1 / total_num_token \sum_{sent ∈ corpus} \sum_{token position ∈ sent} p(lang | token position)

    # every token position is a separate sample
    layer2allentropies = defaultdict(list)
    layer2alltoplang2prob = defaultdict(dict)

    lang_logprobs, lang_token_mask = pickle.load(open(language_distributions_path, "rb"))
    print("lang_logprobs.shape", lang_logprobs.shape)  # [n_langs, vocab_size]
    print("lang_token_mask.shape", lang_token_mask.shape)  # [n_langs, vocab_size]

    total_num_token = 0
    vocab_size = None
    # mask -> true if the token is in the language distribution

    for text in tqdm(data):
        token_logprobs = get_hidden_logprobs(text, tokenizer, model)
        if vocab_size is None:
            vocab_size = token_logprobs.shape[-1]
        device = token_logprobs.device
        lang_logprobs = lang_logprobs.to(device)
        lang_token_mask = lang_token_mask.to(device)

        token_probs = token_logprobs.exp()  # [n_layers, seq_len, vocab_size]
        lang_probs = lang_logprobs.exp().masked_fill(~lang_token_mask, 0)  # [n_lang, vocab_size]
        # if token is not in the language distribution, assign equal probability to all languages
        lang_probs = lang_probs.masked_fill(~(lang_token_mask.any(0)), 1 / lang_probs.shape[0])

        assert token_probs.shape[-1] >= lang_probs.shape[-1]
        if token_probs.shape[-1] > lang_probs.shape[-1]:
            token_probs = token_probs[..., : lang_probs.shape[-1]]
            # renormalize
            token_probs = token_probs / token_probs.sum(-1, keepdim=True)
        token_lang_probs = token_probs @ lang_probs.T  # [n_layers, seq_len, n_langs]
        assert torch.allclose(
            token_lang_probs.sum(dim=-1),
            torch.ones(token_lang_probs.shape[:-1], device=device),  # [n_layers, seq_len]
            atol=0.1,
        )
        seq_len = token_lang_probs.shape[1]
        total_num_token += seq_len

        token_lang_probs = token_lang_probs.cpu()
        layer2entropies, layer2toplangprob = report_and_return(token_lang_probs)
        for layer in layer2entropies:
            layer2allentropies[layer].append(layer2entropies[layer])
        for layer in layer2toplangprob:  # layer2toplangprob[i][LANGUAGES[lang_idx]] = prob
            for language in layer2toplangprob[layer]:
                if language not in layer2alltoplang2prob[layer]:
                    layer2alltoplang2prob[layer][language] = []
                layer2alltoplang2prob[layer][language].append(layer2toplangprob[layer][language])

    layer2lang2prob = defaultdict(dict)
    for layer in layer2alltoplang2prob:
        print(f"Layer {layer}:")
        for language in layer2alltoplang2prob[layer]:
            layer2lang2prob[layer][language] = (
                np.sum(layer2alltoplang2prob[layer][language]) / total_num_token
            )
            print(f"{language} ({layer2lang2prob[layer][language]:.5f})", end="")
            print(", ", end="")
        print("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--language_distributions_path", type=str, required=True)
    parser.add_argument("--lang", type=str, default="zh", choices=LANGUAGES)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=-1)

    args = parser.parse_args()

    tokenizer, model = init_model(args.model_path, args.max_seq_len)
    tokenizer.model_max_length = args.max_seq_len

    data = load_wiki(args.lang, args.num_samples)
    assert len(data) <= args.num_samples, f"data is {len(data)} long"

    calculate_log_prob(data, tokenizer, model, args.language_distributions_path)


if __name__ == "__main__":
    main()
