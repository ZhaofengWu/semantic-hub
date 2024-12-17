import argparse
import copy
import csv
import os
import random
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.logit_lens import init_model, get_emb_and_hidden_states
from lib.utils import batched_cosine_similarity, baseline_adjusted_cosine_similarity
from semantics.logit_lens import parse_instance


def read_cogs(cogs_path, num_samples=10000, filter=False, shuffle=False):
    text_data = []
    formal_data = []
    baseline_data = []

    cnt = 0
    with open(cogs_path, "r") as f:
        tsv_file = csv.reader(f, delimiter="\t")
        for row in tsv_file:
            if cnt >= num_samples:
                break
            if len(row) == 3:  # label, but we won't use
                text = row[0]
                formal = row[1]
                type = row[2]
                if not filter:
                    text_data.append(text)
                    formal_data.append(formal)
                    cnt += 1
                    continue
                if type == "primitive":
                    continue

                name2role = parse_instance(text, formal, ignore_start=False)
                # filtering logic
                if len(name2role) < 2 or len(name2role) > 2:
                    continue

                text_data.append(text)
                formal_data.append(formal)
                # perturbed formal
                names = list(name2role.keys())
                copied_formal = copy.deepcopy(formal)
                name_1 = names[0]
                name_2 = names[1]
                # swap the two names
                copied_formal = copied_formal.replace(name_1, "__temp__")
                copied_formal = copied_formal.replace(name_2, name_1)
                copied_formal = copied_formal.replace("__temp__", name_2)
                if shuffle:
                    copied_formal = shuffle_formal(copied_formal)
                baseline_data.append(copied_formal)
                cnt += 1
    print(f"{cnt} samples")
    return text_data, formal_data, baseline_data


def shuffle_formal(semantics):
    if ";" in semantics:
        lhs, rhs = semantics.rsplit(" ; ", 1)
    else:
        lhs, rhs = "", semantics
    stmts = rhs.split(" AND ")
    random.shuffle(stmts)
    if len(lhs) == 0:
        return " AND ".join(stmts)
    else:
        return lhs + " ; " + " AND ".join(stmts)


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
    parser.add_argument("--filter_data", action="store_true")
    parser.add_argument("--shuffle_data", action="store_true")
    args = parser.parse_args()

    if args.shuffle_data:
        assert args.filter_data

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    tokenizer, model = init_model(args.model_path, args.max_seq_len)
    english_data, non_en_data, baseline_data = read_cogs(
        args.data_path, args.num_samples, filter=args.filter_data, shuffle=args.shuffle_data
    )

    english_hiddens = compute_hiddens(english_data, tokenizer, model)
    non_en_hiddens = compute_hiddens(non_en_data, tokenizer, model)
    if not args.filter_data:
        # cosine_sim: (n_instances, n_instances, n_layers)
        cosine_sim = batched_cosine_similarity(english_hiddens, non_en_hiddens).cpu().float()
        sims_adjusted = baseline_adjusted_cosine_similarity(cosine_sim)
        print(sims_adjusted.mean(1))
    else:
        # orig_sims: (n_instances, n_layers)
        orig_sims = F.cosine_similarity(english_hiddens, non_en_hiddens, dim=-1)
        baseline_sims = F.cosine_similarity(
            english_hiddens, compute_hiddens(baseline_data, tokenizer, model), dim=-1
        )
        print((orig_sims - baseline_sims).mean(0))


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
