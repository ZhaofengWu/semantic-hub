from collections import defaultdict
import math
import os
import pickle
import sys

import inflect
from tqdm import tqdm
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.logit_lens import init_model, get_hidden_logprobs, first_token_id_for_string

inflect_engine = inflect.engine()


def create_data():
    data = []
    for target in range(1, 10):
        # sum_ = other + target
        for other in range(1, 100 - target):
            sum_ = other + target
            assert sum_ < 100
            data.append(("+", other, target, sum_))
        # prod = other * target
        for other in range(1, int(math.ceil(100 / target))):
            prod = other * target
            assert prod < 100
            data.append(("*", other, target, prod))
    return data


CACHE_FILE = "arithmetic_cache_new_new.pkl"


@torch.inference_mode()
def main():
    data = create_data()
    num_probs = defaultdict(list)
    text_probs = defaultdict(list)
    cache = {}
    if os.path.exists(CACHE_FILE):
        cache = pickle.load(open(CACHE_FILE, "rb"))

    for model_name, model_path in [
        ("Llama-2", "meta-llama/Llama-2-7b-hf"),
        ("Llama-3", "meta-llama/Meta-Llama-3-8B"),
    ]:
        model_cache = cache.get(model_path, {})
        tokenizer, model = init_model(model_path)
        curr_cache = model_cache.get(False, {})
        for op, other, target, result in tqdm(data):
            target_text = inflect_engine.number_to_words(target)
            text = f"{result}={other}{op}"
            if text in curr_cache:
                num_prob, text_prob = curr_cache[text]
            else:
                logprobs = get_hidden_logprobs(text, tokenizer, model).cpu()
                num_token = (
                    first_token_id_for_string(tokenizer, str(target), add_prefix_space=False),
                )
                text_token = (
                    first_token_id_for_string(tokenizer, target_text, add_prefix_space=True),
                )
                num_prob = logprobs[:, -1, num_token].tolist()
                text_prob = logprobs[:, -1, text_token].tolist()
                curr_cache[text] = (num_prob, text_prob)
            # The False's are for backward compatibility
            num_probs[(model_name, False)].append(torch.tensor(num_prob))
            text_probs[(model_name, False)].append(torch.tensor(text_prob))
            model_cache[False] = curr_cache
        cache[model_path] = model_cache

    pickle.dump(cache, open(CACHE_FILE, "wb"))

    for model_name in ["Llama-2", "Llama-3"]:
        print(model_name, "Number", torch.stack(num_probs[(model_name, False)], dim=0).mean(0))
        print(model_name, "Text", torch.stack(text_probs[(model_name, False)], dim=0).mean(0))


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
