from collections import defaultdict
import random
import os
import pickle
import sys

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arithmetic.logit_lens import create_data, inflect_engine
from lib.intervention_utils_pyvene import intervened_generate
from lib.logit_lens import init_model, get_emb_and_hidden_states


random.seed(0)


cache_file = "arithmetic_intervention_cache_new.pkl"


def main():
    strengths = [0, 0.5, 1, 1.5, 2, 2.5, 3]

    data = create_data()

    cache = {}
    if os.path.exists(cache_file):
        cache = pickle.load(open(cache_file, "rb"))

    n_correct = {model_name: defaultdict(int) for model_name in ["Llama-2", "Llama-3"]}
    n_unchanged = {model_name: defaultdict(int) for model_name in ["Llama-2", "Llama-3"]}
    n_other = {model_name: defaultdict(int) for model_name in ["Llama-2", "Llama-3"]}
    data = [(op, other, target, result) for op, other, target, result in data if op == "+"]
    for model_name, model_path, layer in [
        ("Llama-2", "meta-llama/Llama-2-7b-hf", 30),
        ("Llama-3", "meta-llama/Meta-Llama-3-8B", 25),
    ]:
        tokenizer, model = init_model(model_path)
        model_cache = cache.get(model_path, {})

        generation_config = dict(max_new_tokens=16, do_sample=False)
        if model_path == "meta-llama/Meta-Llama-3-8B":
            generation_config["pad_token_id"] = tokenizer.eos_token_id

        total = 0
        for op, other, target, result in tqdm(data):
            text = f"{result}={other}{op}"
            pos_token = f"{inflect_engine.number_to_words(result)} equals to {inflect_engine.number_to_words(other + 1)} plus"
            neg_token = f"{inflect_engine.number_to_words(result)} equals to {inflect_engine.number_to_words(other)} plus"
            pos_hidden = get_emb_and_hidden_states(pos_token, tokenizer, model)[layer, -1, :]
            neg_hidden = get_emb_and_hidden_states(neg_token, tokenizer, model)[layer, -1, :]
            total += 1

            for strength in strengths:
                if (text, pos_token, neg_token, layer, strength) in model_cache:
                    intervened_output = model_cache[(text, pos_token, neg_token, layer, strength)]
                else:
                    intervened_output = intervened_generate(
                        tokenizer,
                        model,
                        text,
                        pos_hidden,
                        neg_hidden,
                        strength,
                        layer,
                        -1,
                        generation_config,
                    )
                    model_cache[(text, pos_token, neg_token, layer, strength)] = intervened_output
                if intervened_output.startswith(text + str(target)):
                    n_unchanged[model_name][(layer, strength)] += 1
                elif intervened_output.startswith(text + str(target - 1) + "+1"):
                    n_correct[model_name][(layer, strength)] += 1
                else:
                    n_other[model_name][(layer, strength)] += 1
        cache[model_path] = model_cache

    # reload cache for better concurrency
    old_cache = {}
    if os.path.exists(cache_file):
        old_cache = pickle.load(open(cache_file, "rb"))
    for model_name in ["meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3-8B"]:
        if model_name in old_cache:
            old_cache[model_name].update(cache[model_name])
        else:
            old_cache[model_name] = cache[model_name]
    pickle.dump(old_cache, open(cache_file, "wb"))

    for model_name, layer in [("Llama-2", 30), ("Llama-3", 25)]:
        print(
            model_name,
            "Correct",
            [n_correct[model_name][(layer, strength)] / total * 100 for strength in strengths],
        )
        print(
            model_name,
            "Unchanged",
            [n_unchanged[model_name][(layer, strength)] / total * 100 for strength in strengths],
        )
        print(
            model_name,
            "Other",
            [n_other[model_name][(layer, strength)] / total * 100 for strength in strengths],
        )


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
