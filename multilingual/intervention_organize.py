from collections import defaultdict
import json
import random
import math
import sys
import os

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multilingual.intervention import get_steering_words

random.seed(0)


def read_file(fname):
    with open(fname, "r") as f:
        lines = json.load(f)

    all_scores = defaultdict(list)
    count = 0
    for line in lines:
        empty = len(line["content"].strip()) == 0
        if empty or math.isnan(line["perplexity"]) or math.isinf(line["perplexity"]):
            continue
        all_scores["sentiment_score"].append(line["sentiment"])
        all_scores["perp"].append(line["perplexity"])
        all_scores["rel"].append(line["relevance"])
        count += 1

    avg_scores = {k: sum(v) / count for k, v in all_scores.items()}
    return avg_scores


def read_files_for_seeds(subfolder, layer, coeff, prompt_add, prompt_sub, num_seeds=10):
    all_scores = defaultdict(list)
    for seed in range(num_seeds):
        scores = read_file(
            os.path.join(
                subfolder,
                f"n=1000_l={layer}_c={coeff}_seed={seed}_{prompt_add}_{prompt_sub}_outputs.jsonl",
            ),
        )
        for k, v in scores.items():
            all_scores[k].append(v)
    scores_avg = {k: sum(v) / num_seeds for k, v in all_scores.items()}
    scores_std = {k: np.std(v) for k, v in all_scores.items()}
    return scores_avg, scores_std


def format(avg, std):
    return f"{avg['sentiment']:.3f}$_{{\\pm{std['sentiment']:.3f}}}$ & {avg['perp']:.2f}$_{{\\pm{std['perp']:.2f}}}$ & {avg['rel']:.3f}$_{{\\pm{std['rel']:.3f}}}$"


def report(folder, model, layer, coeff, output_lg):
    subfolder = os.path.join(folder, f"{model}_sentiment")

    prompt_add, prompt_sub = get_steering_words("en", "pos")
    baseline_scores, baseline_scores_std = read_files_for_seeds(
        subfolder, layer, 0, prompt_add, prompt_sub
    )
    en_scores, en_scores_std = read_files_for_seeds(subfolder, layer, coeff, prompt_add, prompt_sub)
    en_rev_scores, en_rev_scores_std = read_files_for_seeds(
        subfolder, layer, coeff, prompt_sub, prompt_add
    )
    prompt_add, prompt_sub = get_steering_words(output_lg, "pos")
    lg_scores, lg_scores_std = read_files_for_seeds(subfolder, layer, coeff, prompt_add, prompt_sub)
    lg_rev_scores, lg_rev_scores_std = read_files_for_seeds(
        subfolder, layer, coeff, prompt_sub, prompt_add
    )
    print(model)
    print(f"{format(baseline_scores, baseline_scores_std)}")
    print(f"{format(lg_scores, lg_scores_std)}")
    print(f"{format(en_scores, en_scores_std)}")
    print(f"{format(lg_rev_scores, lg_rev_scores_std)}")
    print(f"{format(en_rev_scores, en_rev_scores_std)}")


def main():
    for lg in ["es", "zh"]:
        for model in ["llama2", "llama3"]:
            report(f"actadd_new_prompted{lg}", model, 17, 2, lg)


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
