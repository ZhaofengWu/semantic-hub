import ast
from collections import defaultdict
import random
import os
import pickle
import sys

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coding.semantic_role import load_mbpp, get_character_offsets
from lib.intervention_utils_pyvene import intervened_generate
from lib.logit_lens import init_model, get_unembedding


random.seed(0)


def load_data():
    programs = load_mbpp()
    prompts = []
    for program in programs:
        tree = ast.parse(program)
        # find all calls to `range`
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "range"
                and len(node.args) == 1
            ):
                start, end = get_character_offsets(program, node)
                if program.startswith(program[:start] + "range("):
                    prompt = program[:start] + "range("
                elif program.startswith(program[:start] + "range ("):
                    prompt = program[:start] + "range ("
                else:
                    assert False
                prompts.append(prompt)
    return prompts


def arguments(prompt, generated_text):
    assert generated_text.startswith(prompt)
    generation = generated_text[len(prompt) :]
    if generation.strip()[0] == ")":
        return []
    arguments = []
    buffer = []
    depth = 1
    # find the argument list
    for c in generation:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        if depth == 0:
            arguments.append("".join(buffer).strip())
            buffer = []
            break
        if c == "," and depth == 1:
            arguments.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(c)
    return arguments


cache_file = "actadd_code_cache.pkl"


def main():
    model_name = "meta-llama/Llama-2-7b-hf"
    pos_token = "start"
    neg_token = "end"
    layer = 17
    strengths = [10, 20, 30, 40, 50, 60, 70, 80]

    prompts = load_data()
    tokenizer, model = init_model(model_name)
    pos_hidden = get_unembedding(pos_token, model, tokenizer)
    neg_hidden = get_unembedding(neg_token, model, tokenizer)

    generation_config = dict(max_new_tokens=32, do_sample=False)
    if model_name == "meta-llama/Meta-Llama-3-8B":
        generation_config["pad_token_id"] = tokenizer.eos_token_id

    n_correct = defaultdict(int)
    n_unchanged = defaultdict(int)
    n_other = defaultdict(int)
    cache = {}
    if os.path.exists(cache_file):
        cache = pickle.load(open(cache_file, "rb"))
    for prompt in tqdm(prompts):
        output = intervened_generate(
            tokenizer,
            model,
            prompt,
            pos_hidden,
            neg_hidden,
            0,
            layer,
            -1,
            generation_config,
        )

        default_predicted_args = arguments(prompt, output)
        if len(default_predicted_args) != 1:
            continue

        for strength in strengths:
            if (layer, strength) in cache:
                continue

            intervened_output = intervened_generate(
                tokenizer,
                model,
                prompt,
                pos_hidden,
                neg_hidden,
                strength,
                layer,
                -1,
                generation_config,
            )
            intervened_predicted_args = arguments(prompt, intervened_output)

            if ["0"] + default_predicted_args == intervened_predicted_args:
                n_correct[(layer, strength)] += 1
            elif default_predicted_args == intervened_predicted_args:
                n_unchanged[(layer, strength)] += 1
            else:
                n_other[(layer, strength)] += 1

    for strength in strengths:
        if (layer, strength) in cache:
            (
                n_correct[(layer, strength)],
                tmp,
                n_unchanged[(layer, strength)],
                n_other[(layer, strength)],
            ) = cache[(layer, strength)]
            n_other[(layer, strength)] += tmp
        else:
            # 0 is for backward compatibility
            cache[(layer, strength)] = (
                n_correct[(layer, strength)],
                0,
                n_unchanged[(layer, strength)],
                n_other[(layer, strength)],
            )
    pickle.dump(cache, open(cache_file, "wb"))

    corrects = [n_correct[(layer, strength)] for strength in strengths]
    unchangeds = [n_unchanged[(layer, strength)] for strength in strengths]
    others = [n_other[(layer, strength)] for strength in strengths]
    totals = [
        correct + unchanged + other
        for correct, unchanged, other in zip(corrects, unchangeds, others)
    ]
    assert all(total == totals[0] for total in totals)
    total = totals[0]

    print("Correct", [0] + [n / total * 100 for n in corrects])
    print("Unchanged", [100] + [n / total * 100 for n in unchangeds])
    print("Other", [0] + [n / total * 100 for n in others])


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
