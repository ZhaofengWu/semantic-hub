from collections import defaultdict
import os
import re
import pickle
import sys

import numpy as np
import pyvene as pv
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.intervention_utils_pyvene import intervened_generate
from lib.logit_lens import init_model, get_unembedding

CACHE_FILE = "image_intervention_cache.pkl"
COLORS = {"red", "green", "blue", "black"}
MAPPINGS = {
    "red": "#FF000",
    "green": "#00FF00",
    "blue": "#0000FF",
    "black": "#000000",
}


def half_image_with_color(color):
    image_tensor = torch.zeros(3, 128, 256)
    if color == "red":
        image_tensor[0] = 1
    if color == "green":
        image_tensor[1] = 1
    if color == "blue":
        image_tensor[2] = 1
    return image_tensor


def parse_output(output):
    if output.count("?") != 1:
        return None
    answer = output.split("?", maxsplit=1)[1].split(".")[0].strip()
    # match all "color1 and color2"
    matches = set(re.findall(r"((\w+) and (\w+))", answer))
    if len(matches) != 1:
        return None
    match = matches.pop()
    return {match[1], match[2]}


def do_one_image(
    color_top,
    color_bottom,
    color_new,
    side,
    model,
    processor,
    target_layer,
    strength,
    generation_config,
    intervention_type,
):
    assert all(color in COLORS for color in [color_top, color_bottom, color_new])
    assert color_top != color_bottom
    image_tensor = torch.cat(
        [half_image_with_color(color_top), half_image_with_color(color_bottom)], dim=1
    )
    image = transforms.ToPILImage()(image_tensor)
    prompt = "What are the two colors in the image?\n<image>"
    prefix_tokenized = processor(prompt, image, return_tensors="pt").to(
        device=model.device, dtype=model.dtype
    )

    image_token = processor.tokenizer.convert_tokens_to_ids("<image>")
    image_token_idx = (prefix_tokenized["input_ids"] == image_token).nonzero(as_tuple=True)[1]
    assert len(image_token_idx) == 1024
    position = image_token_idx.tolist()
    if side == "top":
        position = position[: len(position) // 2]
    elif side == "bottom":
        position = position[len(position) // 2 :]
    else:
        raise ValueError(f"side must be top or bottom, got {side}")

    color_old = color_top if side == "top" else color_bottom
    pos_hidden = get_unembedding(color_new, model, processor.tokenizer)
    neg_hidden = get_unembedding(color_old, model, processor.tokenizer)
    intervened_output = intervened_generate(
        processor.tokenizer,
        model,
        prefix_tokenized,
        pos_hidden,
        neg_hidden,
        strength,
        target_layer,
        position,
        generation_config,
        intervention_type=intervention_type,
    )
    colors = parse_output(intervened_output)
    if colors is None:
        return "unparseable"
    if colors == colors - {color_old} | {color_new}:
        return "correct"
    else:
        return "other"


@torch.inference_mode()
def main(model_path):
    assert "chameleon" in model_path.lower()
    model = processor = generation_config = None

    target_layers = [tuple(range(i, 32)) for i in range(32)]
    strength = 1
    intervention_type = pv.VanillaIntervention

    if os.path.exists(CACHE_FILE):
        cache = pickle.load(open(CACHE_FILE, "rb"))
    else:
        cache = {}
    for target_layer in tqdm(target_layers):
        if (target_layer, strength, intervention_type) in cache:
            print(f"target {target_layer}, strength {strength}")
            print(cache[(target_layer, strength, intervention_type)])
            continue
        if model is None:
            processor, model = init_model(model_path)
            generation_config = dict(
                max_new_tokens=15,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        results = defaultdict(int)
        for color_top in COLORS:
            for color_bottom in COLORS:
                if color_top == color_bottom:
                    continue
                for color_new in COLORS:
                    if color_new in {color_top, color_bottom}:
                        continue
                    for side in ["top", "bottom"]:
                        result = do_one_image(
                            color_top,
                            color_bottom,
                            color_new,
                            side,
                            model,
                            processor,
                            target_layer,
                            strength,
                            generation_config,
                            intervention_type,
                        )
                        results[result] += 1
        cache[(target_layer, strength, intervention_type)] = results
        pickle.dump(cache, open(CACHE_FILE, "wb"))

    results = [cache[tuple(range(l, 32)), strength, intervention_type] for l in range(32)]
    total = 48
    correct = np.array(
        [[1] * r["correct"] + [0] * (total - r["correct"]) for r in results]
    )  # (num_layers, bsz)
    accs = correct.mean(1)
    print(accs * 100)


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
