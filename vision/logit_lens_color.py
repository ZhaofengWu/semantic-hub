import random
import os
import sys

import torch
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.logit_lens import init_model, get_llava_hidden_logprobs

random.seed(1)


def compute_logprobs(image, model, processor):
    prompt = "USER: What is the color in the image?<image>\n ASSISTANT:"

    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
    inputs["labels"] = inputs["input_ids"]
    log_probs, ids = get_llava_hidden_logprobs(
        inputs, model
    )  # logprobs: (n_layers, n_tokens, vocab_size)
    image_tokens = (ids == 32000).squeeze(0)
    image_logprobs = log_probs[:, image_tokens]  # (n_layers, n_image_tokens, vocab_size)
    return image_logprobs


def main(model_path):
    processor, model = init_model(model_path)

    colors = ["red", "green", "blue", "black"]
    tops = []
    white_tops = []
    for color in tqdm(colors):
        image_tensor = torch.zeros(3, 256, 256)
        if color != "black":
            image_tensor[colors.index(color)] = 1
        to_pil = transforms.ToPILImage()
        image = to_pil(image_tensor)
        logprobs = compute_logprobs(image, model, processor)

        label_token_id = processor.tokenizer(color, add_special_tokens=False)["input_ids"][0]
        label_ranks = (
            (logprobs.argsort(descending=True, dim=-1) == label_token_id)
            .nonzero()[:, 2]
            .reshape(*logprobs.shape[:2])
        )
        tops.append((label_ranks == 0).float().cpu())

        white_token_id = processor.tokenizer("white", add_special_tokens=False)["input_ids"][0]
        white_ranks = (
            (logprobs.argsort(descending=True, dim=-1) == white_token_id)
            .nonzero()[:, 2]
            .reshape(*logprobs.shape[:2])
        )
        white_tops.append((white_ranks == 0).float().cpu())

    tops = torch.cat(tops, dim=-1)  # (n_layers, n_colors * n_tokens)
    white_tops = torch.cat(white_tops, dim=-1)

    print("Corresponding Color", tops.mean(1) * 100)
    print("White", white_tops.mean(1) * 100)


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
