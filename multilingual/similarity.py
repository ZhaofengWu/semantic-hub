import argparse
import csv
import os
import random
import sys

import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.logit_lens import init_model, get_emb_and_hidden_states
from lib.utils import batched_cosine_similarity, baseline_adjusted_cosine_similarity


def read_gale_files(gale_data_path, num_samples=10000):
    source_file_dir = os.path.join(gale_data_path, "source")  # zh
    translation_file_dir = os.path.join(gale_data_path, "translation")  # en
    source_data = []
    translation_data = []
    source_files = os.listdir(source_file_dir)
    cnt = 0
    for source_file in source_files:
        if cnt >= num_samples:
            break
        translation_file = source_file.replace(".tdf", ".eng.tdf")
        source_file_path = os.path.join(source_file_dir, source_file)
        translation_file_path = os.path.join(translation_file_dir, translation_file)
        with open(source_file_path) as f_source:
            with open(translation_file_path) as f_trans:
                source_tsv_file = csv.DictReader(f_source, delimiter="\t")
                translate_tsv_file = csv.DictReader(f_trans, delimiter="\t")
                for source_row, translate_row in zip(source_tsv_file, translate_tsv_file):
                    if cnt >= num_samples:
                        break
                    if (
                        source_row["transcript;unicode"] is None
                        or translate_row["transcript;unicode"] is None
                        or len(translate_row["transcript;unicode"]) == 0
                        or len(source_row["transcript;unicode"]) == 0
                    ):
                        continue
                    else:
                        source_data.append(source_row["transcript;unicode"])
                        translation_data.append(translate_row["transcript;unicode"])
                    cnt += 1

        assert len(source_data) == len(
            translation_data
        ), f"translation is length {len(translation_data)} while source is {len(source_data)}"
    return source_data, translation_data


def augment_data(chinese_data, english_data):
    augmented_chinese_data = []
    for chinese_text in chinese_data:
        if chinese_text.endswith("。"):
            chinese_text += "这代表了"
            augmented_chinese_data.append(chinese_text)
        else:
            augmented_chinese_data.append(chinese_text)
    augmented_english_data = []
    for english_text in english_data:
        if english_text.endswith(".") or english_text.endswith('."'):
            english_text += "This represents"
            augmented_english_data.append(english_text)
        else:
            augmented_english_data.append(english_text)
    return augmented_chinese_data, augmented_english_data


def compute_hiddens(texts, tokenizer, model):
    return torch.stack(
        [get_emb_and_hidden_states(text, tokenizer, model)[1:, -1, :] for text in tqdm(texts)],
        dim=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    args.num_samples = int(args.num_samples)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    tokenizer, model = init_model(args.model_path, args.max_seq_len)
    non_en_data, english_data = read_gale_files(args.data_path, args.num_samples)
    non_en_data, english_data = augment_data(non_en_data, english_data)

    english_hiddens = compute_hiddens(english_data, tokenizer, model)
    non_en_hiddens = compute_hiddens(non_en_data, tokenizer, model)
    cosine_sim = batched_cosine_similarity(english_hiddens, non_en_hiddens).cpu().float()
    sims_adjusted = baseline_adjusted_cosine_similarity(cosine_sim)
    print(sims_adjusted.mean(1))


if __name__ == "__main__":
    try:
        main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter,too-many-function-args
    except Exception as e:
        import pdb
        import traceback

        if not isinstance(e, (pdb.bdb.BdbQuit, KeyboardInterrupt)):
            print("\n" + ">" * 100 + "\n")
            traceback.print_exc()
            print()
            pdb.post_mortem()
