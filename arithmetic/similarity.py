import os
import pickle
import sys

import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arithmetic.logit_lens import create_data, inflect_engine
from lib.logit_lens import init_model, get_emb_and_hidden_states
from lib.utils import batched_cosine_similarity


def compute_hiddens(text, model, tokenizer):
    return get_emb_and_hidden_states(text, tokenizer, model)[1:, -1, :]


def compute_cosine_similarities(model_path, cache_file, numbers, texts):
    if os.path.exists(cache_file):
        cosine_sim = pickle.load(open(cache_file, "rb"))
    else:
        tokenizer, model = init_model(model_path)
        numbers_hiddens = [compute_hiddens(program, model, tokenizer) for program in tqdm(numbers)]
        texts_hiddens = [compute_hiddens(instr, model, tokenizer) for instr in tqdm(texts)]

        numbers_hiddens = torch.stack(numbers_hiddens, dim=0)  # (n_numbers, n_layers, hidden_size)
        texts_hiddens = torch.stack(texts_hiddens, dim=0)  # (n_texts, n_layers, hidden_size)
        cosine_sim = batched_cosine_similarity(
            numbers_hiddens, texts_hiddens
        )  # (n_numbers, n_texts, n_layers)
        cosine_sim = cosine_sim.cpu().float()
    return cosine_sim


def main():
    data = create_data()
    numbers = [f"{other}{op}{target}" for op, other, target, _ in data]
    op_map = {"+": "plus", "*": "times"}
    texts = [
        f"{inflect_engine.number_to_words(other)} {op_map[op]} {inflect_engine.number_to_words(target)}"
        for op, other, target, _ in data
    ]
    same_result_same_op_mask = [
        [result1 == result2 and op1 == op2 for op1, _, _, result1 in data]
        for op2, _, _, result2 in data
    ]
    same_result_same_op_mask = torch.tensor(same_result_same_op_mask, dtype=torch.bool)
    same_result_diff_op_mask = [
        [result1 == result2 and op1 != op2 for op1, _, _, result1 in data]
        for op2, _, _, result2 in data
    ]
    same_result_diff_op_mask = torch.tensor(same_result_diff_op_mask, dtype=torch.bool)
    diff_result_same_op_mask = [
        [result1 != result2 and op1 == op2 for op1, _, _, result1 in data]
        for op2, _, _, result2 in data
    ]
    diff_result_same_op_mask = torch.tensor(diff_result_same_op_mask, dtype=torch.bool)
    diff_result_diff_op_mask = [
        [result1 != result2 and op1 != op2 for op1, _, _, result1 in data]
        for op2, _, _, result2 in data
    ]
    diff_result_diff_op_mask = torch.tensor(diff_result_diff_op_mask, dtype=torch.bool)

    model_to_cosine_sim = {}

    for model_name, model_path, cache_path in [
        ("Llama-2", "meta-llama/Llama-2-7b-hf", "arithmetic_repr_cache_llama2.pkl"),
        ("Llama-3", "meta-llama/Meta-Llama-3-8B", "arithmetic_repr_cache_llama3.pkl"),
    ]:
        cosine_sim = compute_cosine_similarities(model_path, cache_path, numbers, texts)
        model_to_cosine_sim[model_name] = cosine_sim

    for model_name, cosine_sim in model_to_cosine_sim.items():
        diag_cosine_sim = torch.diagonal(cosine_sim, dim1=0, dim2=1)  # (n_layers, n_images)
        same_result_same_op_sim = cosine_sim[same_result_same_op_mask].transpose(
            0, 1
        )  # (n_layers, n_same_result_same_op)
        same_result_diff_op_sim = cosine_sim[same_result_diff_op_mask].transpose(
            0, 1
        )  # (n_layers, n_same_result_diff_op)
        diff_result_same_op_sim = cosine_sim[diff_result_same_op_mask].transpose(
            0, 1
        )  # (n_layers, n_diff_result_same_op)
        diff_result_diff_op_sim = cosine_sim[diff_result_diff_op_mask].transpose(
            0, 1
        )  # (n_layers, n_diff_result_diff_op)
        print(model_name, "Exact match", diag_cosine_sim.mean(1))
        print(
            model_name,
            "Same result",
            torch.cat([same_result_same_op_sim, same_result_diff_op_sim], dim=1).mean(1),
        )
        print(
            model_name,
            "Diff result",
            torch.cat([diff_result_same_op_sim, diff_result_diff_op_sim], dim=1).mean(1),
        )

        baseline = torch.cat(
            [
                same_result_same_op_sim,
                same_result_diff_op_sim,
                diff_result_same_op_sim,
                diff_result_diff_op_sim,
            ],
            dim=1,
        )
        print(model_name, "Diff", (diag_cosine_sim - baseline.mean(-1, keepdim=True)).mean(1))


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
