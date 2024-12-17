import argparse
from collections import defaultdict
import os
import re
import sys
from tqdm import tqdm

import jieba
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.logit_lens import init_model, get_hidden_logprobs
from compute_percentages import load_wiki


def parse_zhen_dict(file):
    zh2en = defaultdict(list)
    with open(file) as fin:
        for line in fin:
            if line == "":
                continue
            line = line.rstrip("/")
            line = line.split("/")
            if len(line) <= 1:
                continue
            english = line[1]
            if "surname" in english:
                continue
            english = re.sub("\(.*?\)", "()", english)
            english = english.replace("(", "").replace(")", "").strip()
            char_and_pinyin = line[0].split("[")
            characters = char_and_pinyin[0]
            characters = characters.split()
            simplified = characters[1]
            zh2en[simplified].append(english)
    return zh2en


def jieba_segment(text):
    segmented = list(jieba.cut(text, cut_all=False))
    token_start = 0
    jieba_mapping = []
    for segment in segmented:
        token_end = token_start + len(segment)  # Calculate the end position
        jieba_mapping.append((token_start, token_end))
        token_start = token_end
    return segmented, jieba_mapping


def get_indices_in_intersection(original_mapping, jieba_mapping, intersection):
    original_indices = []
    jieba_indices = []

    # Find indices in original_mapping
    for i, value in enumerate(original_mapping):
        if value in intersection:
            original_indices.append(i)

    # Find indices in jieba_mapping
    for i, value in enumerate(jieba_mapping):
        if value in intersection:
            jieba_indices.append(i)

    return original_indices, jieba_indices


def cut_off_text(tokenizer, text, orig2trans):
    original = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    original_encoded = original["input_ids"]
    original_mapping = original["offset_mapping"]
    if len(tokenizer.decode(original_encoded[0])) == 0:  # for llama2's space
        original_encoded.pop(0)
        original_mapping.pop(0)
    originalmap2idx = {original_mapping[idx]: idx for idx in range(len(original_mapping))}
    segments, mapping = jieba_segment(text)
    translated = [orig2trans[text] for text in segments]

    intersection = sorted(list(set(original_mapping) & set(mapping)), key=lambda x: x[0])
    # get index of the mapping
    original_indices, segment_indices = get_indices_in_intersection(
        original_mapping, mapping, intersection
    )

    # index of the original sequence to encoded translation
    # where the translation is the token's next => index needs to -1
    previdx2encoded_translation = dict()
    # previous token to next token in the original sequence
    previdx2next_tok = dict()

    # iterating original_indices guarantees each tokenized element to be a single token
    for i in range(len(original_indices)):
        map_idx = original_indices[i]
        segment_map_idx = segment_indices[i]
        curr_idx = originalmap2idx[original_mapping[map_idx]]
        if curr_idx != map_idx:
            return None, None, None
        if original_mapping[map_idx] != mapping[segment_map_idx]:
            return None, None, None
        curr_translations = translated[segment_map_idx]

        # segmentation is a single token (by tokenizer) but translation is emtpy
        if len(curr_translations) == 0:
            continue
        if len(curr_translations) == 1:  # for iterating purposes
            curr_translations.append("")

        # check whether eligible translation is available (at least encoded one single-token translation)
        eligible_translations = []
        for translation in curr_translations:
            if len(translation) == 0:
                continue
            if "llama3" in tokenizer.name_or_path.lower().replace("-", ""):
                translation = " " + translation
            encoded_trans = tokenizer.encode(translation, add_special_tokens=False)
            if len(encoded_trans) > 1:  # multiple-token translation
                continue
            elif len(encoded_trans) == 0:  # empty translation
                continue
            else:  # single-token translation
                eligible_translations.append(encoded_trans[0])

        if len(eligible_translations) == 0:
            continue

        prev_idx = map_idx - 1
        if prev_idx < 0:
            continue

        previdx2encoded_translation[prev_idx] = eligible_translations
        previdx2next_tok[prev_idx] = original_encoded[map_idx]

    return previdx2encoded_translation, previdx2next_tok, original_encoded


def segment_text(text, orig2trans, tokenizer, model):
    previdx2encoded_translation, previdx2next_tok, _ = cut_off_text(tokenizer, text, orig2trans)
    if (
        previdx2encoded_translation is None
        or previdx2next_tok is None
        or len(previdx2encoded_translation) == 0
        or len(previdx2next_tok) == 0
    ):
        return None, None

    token_logprobs = get_hidden_logprobs(text, tokenizer, model, add_special_tokens=False)
    layer2tokidx2max_prob_token = defaultdict(dict)
    layer2tokidx2next_token_prob = defaultdict(dict)
    # for each layer
    for layer_i, layer in enumerate(token_logprobs):  # (seq_len, vocab size)
        # for each prev idx
        for prev_idx in previdx2encoded_translation:  # which token has eligible translation
            assert prev_idx in previdx2next_tok, f"index {prev_idx} not in previdx2next_tok"
            # encoded next token translation
            next_translated_token_encoded = previdx2encoded_translation[prev_idx]
            # get actual probs
            next_actual_token_id = previdx2next_tok[prev_idx]
            next_actual_token_prob = layer[prev_idx, next_actual_token_id]
            # initialize
            max_prob = -torch.inf
            max_token_id = None
            # max_{translation \in translations} p^{logitlens}(translation | h)
            for translate_token_id in next_translated_token_encoded:
                # check the prob of the next translated token
                next_translated_token_prob = layer[prev_idx, translate_token_id]
                # update the max prob and the token id
                if next_translated_token_prob > max_prob:
                    max_prob = next_translated_token_prob
                    max_token_id = translate_token_id

            if prev_idx not in layer2tokidx2max_prob_token[layer_i]:
                layer2tokidx2max_prob_token[layer_i][prev_idx] = dict()
            if prev_idx not in layer2tokidx2next_token_prob[layer_i]:
                layer2tokidx2next_token_prob[layer_i][prev_idx] = dict()
            layer2tokidx2max_prob_token[layer_i][prev_idx] = (max_prob.item(), max_token_id)
            layer2tokidx2next_token_prob[layer_i][prev_idx] = (
                next_actual_token_prob.item(),
                next_actual_token_id,
            )

    return layer2tokidx2max_prob_token, layer2tokidx2next_token_prob


def aggregate_probs(curr_dict):
    total = 0
    cnt = 0
    for prev_idx in curr_dict:
        for token_prob, token_id in curr_dict[prev_idx]:
            total += token_prob
            cnt += 1
    return total, cnt


def compute(data, tokenizer, model, orig2trans):
    total_layer2tokidx2max_prob_token = defaultdict(dict)
    total_layer2tokidx2next_token_prob = defaultdict(dict)
    for text in tqdm(data):
        layer2tokidx2max_prob_token, layer2tokidx2next_token_prob = segment_text(
            text, orig2trans, tokenizer, model
        )
        if layer2tokidx2max_prob_token is None or layer2tokidx2next_token_prob is None:
            continue
        # aggregate
        for layer_i in layer2tokidx2max_prob_token:
            for prev_idx in layer2tokidx2max_prob_token[layer_i]:
                if prev_idx not in total_layer2tokidx2max_prob_token[layer_i]:
                    total_layer2tokidx2max_prob_token[layer_i][prev_idx] = []
                total_layer2tokidx2max_prob_token[layer_i][prev_idx].append(
                    layer2tokidx2max_prob_token[layer_i][prev_idx]
                )

        for layer_i in layer2tokidx2next_token_prob:
            for prev_idx in layer2tokidx2next_token_prob[layer_i]:
                if prev_idx not in total_layer2tokidx2next_token_prob[layer_i]:
                    total_layer2tokidx2next_token_prob[layer_i][prev_idx] = []
                total_layer2tokidx2next_token_prob[layer_i][prev_idx].append(
                    layer2tokidx2next_token_prob[layer_i][prev_idx]
                )

    # compute the average
    avg_translations = []
    avg_actual_nexts = []

    total_cnt = 0
    for layer_i in total_layer2tokidx2max_prob_token:
        translation_total, translation_cnt = aggregate_probs(
            total_layer2tokidx2max_prob_token[layer_i]
        )
        avg_translations.append(translation_total / translation_cnt)
        actual_total, actual_cnt = aggregate_probs(total_layer2tokidx2next_token_prob[layer_i])
        avg_actual_nexts.append(actual_total / actual_cnt)
        if total_cnt != 0:
            assert total_cnt == translation_cnt == actual_cnt
        else:
            total_cnt = translation_cnt

    print("Chinese true next tokens:", avg_actual_nexts)
    print("English translations:", avg_translations)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dict_path", type=str, default="cedict_ts.u8")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--max_seq_len", type=int, default=2048)

    args = parser.parse_args()
    zh2en = parse_zhen_dict(args.dict_path)
    tokenizer, model = init_model(args.model_path, args.max_seq_len)
    data = load_wiki("zh", args.num_samples)
    compute(data, tokenizer, model, zh2en)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import pdb
        import traceback

        if not isinstance(e, (pdb.bdb.BdbQuit, KeyboardInterrupt)):
            print("\n" + ">" * 100 + "\n")
            traceback.print_exc()
            print()
            pdb.post_mortem()
