import json
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.logit_lens import init_model


@torch.inference_mode()
def compute_relevance(tokenizer, model, prompt, continuation):
    # mostly follow https://huggingface.co/intfloat/multilingual-e5-large
    def compute_embedding(text):
        # peculiarity of intfloat/multilingual-e5-large
        encoded_input = tokenizer(
            "query: " + text, max_length=512, truncation=True, return_tensors="pt"
        ).to(model.device)
        hidden = model(**encoded_input, output_hidden_states=True)
        emb = hidden.hidden_states[-1].mean(1)
        return F.normalize(emb)  # this may not be necessary given cosine similarity?

    prompt_emb = compute_embedding(prompt)
    continuation_emb = compute_embedding(continuation)
    cosine_similarity = F.cosine_similarity(prompt_emb, continuation_emb, dim=1)
    return cosine_similarity.cpu().item()


def main(model_name, *files):
    tokenizer, model = init_model(model_name)
    for file in files:
        lines = json.load(open(file))
        for line in tqdm(lines):
            key = "relevance"
            if key in line:
                continue
            relevance = compute_relevance(tokenizer, model, line["prompt"], line["content"])
            line[key] = relevance
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
