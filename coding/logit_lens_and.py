import ast
import os
import pickle as pkl
import sys

import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coding.semantic_role import load_mbpp, get_character_offsets
from lib.logit_lens import init_model, get_hidden_logprobs, first_token_id_for_string


CACHE_FILE = "and_cache_llama2_mbpp.pkl"


def main():
    model_path = "meta-llama/Llama-2-7b-hf"
    if os.path.exists(CACHE_FILE):
        total_probs_and, total_probs_or, total_probs_not, total_probs_next_token = pkl.load(
            open(CACHE_FILE, "rb")
        )
    else:
        tokenizer, model = init_model(model_path)
        programs = load_mbpp(include_tests=True)
        total_probs_and = []
        total_probs_or = []
        total_probs_not = []
        total_probs_next_token = []
        for program_idx, program in enumerate(tqdm(programs)):
            tree = ast.parse(program)

            lists = [
                node for node in ast.walk(tree) if isinstance(node, ast.List) and len(node.elts) > 1
            ]
            if len(lists) == 0:
                continue

            logprobs = get_hidden_logprobs(program, tokenizer, model).cpu()
            encoded = tokenizer.encode_plus(program, return_offsets_mapping=True)
            input_ids = encoded["input_ids"]
            token_offsets = encoded["offset_mapping"]

            for lst in lists:
                for element in lst.elts:
                    _, end = get_character_offsets(program, element)
                    while program[end] in {" ", "\n"}:
                        end += 1
                    if program[end] not in {",", "]"}:
                        # corner cases
                        continue
                    if program[end] == "]":
                        continue

                    token_indices_that_are_comma = [
                        token_idx
                        for token_idx, (s, e) in enumerate(token_offsets)
                        if s <= end < e and program[s:e].strip(" ") == ","
                    ]
                    assert len(token_indices_that_are_comma) <= 1
                    if len(token_indices_that_are_comma) == 0:
                        # e.g., "'," (quote + comma)
                        continue
                    comma_idx = token_indices_that_are_comma[0]
                    assert tokenizer.convert_ids_to_tokens(input_ids[comma_idx]).strip("▁") == ","

                    and_token_id = first_token_id_for_string(
                        tokenizer, "and", add_prefix_space=True
                    )
                    probs_and = logprobs[:, comma_idx, and_token_id]
                    or_token_id = first_token_id_for_string(tokenizer, "or", add_prefix_space=True)
                    probs_or = logprobs[:, comma_idx, or_token_id]
                    not_token_id = first_token_id_for_string(
                        tokenizer, "not", add_prefix_space=True
                    )
                    probs_not = logprobs[:, comma_idx, not_token_id]
                    probs_next_token = logprobs[:, comma_idx, input_ids[comma_idx + 1]]

                    total_probs_and.append(probs_and)
                    total_probs_or.append(probs_or)
                    total_probs_not.append(probs_not)
                    total_probs_next_token.append(probs_next_token)

        total_probs_and = torch.stack(total_probs_and, dim=0)
        total_probs_or = torch.stack(total_probs_or, dim=0)
        total_probs_not = torch.stack(total_probs_not, dim=0)
        total_probs_next_token = torch.stack(total_probs_next_token, dim=0)
        pkl.dump(
            (total_probs_and, total_probs_or, total_probs_not, total_probs_next_token),
            open(CACHE_FILE, "wb"),
        )

    print("Next token", total_probs_next_token.mean(0))
    print("not", total_probs_not.mean(0))
    print("or", total_probs_or.mean(0))
    print("and", total_probs_and.mean(0))


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
