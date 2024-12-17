import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.logit_lens import init_model


@torch.inference_mode()
def compute_perplexity(tokenizer, model, prompt, continuation):
    def logprob(input_ids):
        logits = model(input_ids, return_dict=True).logits
        assert logits.shape[0] == 1
        logprobs = logits[0, :-1].log_softmax(dim=-1)
        labels = input_ids[0, 1:]
        selected_logprobs = logprobs[torch.arange(len(labels)), labels]
        return selected_logprobs.sum()

    # compute the conditional perplexity of the continuation given the prompt
    prompt_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    combined_inputs = tokenizer(prompt + continuation, return_tensors="pt").to(model.device)

    prompt_ids = prompt_inputs.input_ids
    combined_ids = combined_inputs.input_ids
    prompt_logprob = logprob(prompt_ids)
    combined_logprob = logprob(combined_ids)

    return (
        ((prompt_logprob - combined_logprob) / (combined_ids.shape[1] - prompt_ids.shape[1]))
        .exp()
        .cpu()
        .item()
    )


def main(model_name, *files):
    tokenizer, model = init_model(model_name)
    for file in files:
        lines = json.load(open(file))
        for line in tqdm(lines):
            key = "perplexity"
            if key in line:
                continue
            perplexity = compute_perplexity(tokenizer, model, line["prompt"], line["content"])
            line[key] = perplexity
        json.dump(lines, open(file, "w"), indent=2)


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
