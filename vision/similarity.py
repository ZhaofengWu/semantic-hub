import json
import os
import pickle
import sys
from PIL import Image

import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.logit_lens import init_model, get_emb_and_hidden_states
from lib.modeling_llava import LlavaForConditionalGeneration
from lib.modeling_chameleon import ChameleonForConditionalGeneration
from lib.utils import batched_cosine_similarity, baseline_adjusted_cosine_similarity


def load_data(coco_dir):
    images = {}
    images_dir = os.path.join(coco_dir, "val2017")
    for filename in os.listdir(images_dir):
        image_id = int(filename.split(".")[0])
        image = Image.open(os.path.join(images_dir, filename))
        image.load()
        images[image_id] = image
    data = []
    captions_path = os.path.join(coco_dir, "annotations/captions_val2017.json")
    captions_data = json.load(open(captions_path))
    for item in captions_data["annotations"]:
        image_id = item["image_id"]
        caption = item["caption"]
        data.append((images[image_id], caption))
    return data


def compute_image_hiddens(image, model, processor):
    if isinstance(model, LlavaForConditionalGeneration):
        prompt = "<image>"
    elif isinstance(model, ChameleonForConditionalGeneration):
        prompt = "What is in the image?\n<image>"

    inputs = processor(prompt, image, return_tensors="pt").to(
        device=model.device, dtype=model.dtype
    )
    inputs["labels"] = inputs["input_ids"]
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    all_hidden_states = torch.stack([h.squeeze(0) for h in hidden_states], dim=0)[1:]
    return all_hidden_states[:, -1, :]


def compute_caption_hiddens(caption, model, processor):
    return get_emb_and_hidden_states(caption, processor.tokenizer, model)[1:, -1, :]


@torch.inference_mode()
def main(coco_dir, n=1000):
    model_to_cosine_sim = {}
    for model_name, model_path, cache_file in [
        ("Llava", "llava-hf/llava-1.5-7b-hf", "caption_repr_cache_llava_1000.pkl"),
        ("Chameleon", "facebook/chameleon-7b", "caption_repr_cache_chameleon_1000.pkl"),
    ]:
        if os.path.exists(cache_file):
            cosine_sim = pickle.load(open(cache_file, "rb"))
        else:
            full_data = load_data(coco_dir)
            data = full_data[: int(n)]
            processor, model = init_model(model_path)

            image_hiddens = [
                compute_image_hiddens(image, model, processor).cpu()
                for image, caption in tqdm(data)
            ]
            caption_hiddens = [
                compute_caption_hiddens(caption, model, processor).cpu()
                for image, caption in tqdm(data)
            ]

            image_hiddens = torch.stack(image_hiddens, dim=0)  # (n_images, n_layers, hidden_size)
            caption_hiddens = torch.stack(
                caption_hiddens, dim=0
            )  # (n_captions, n_layers, hidden_size)
            cosine_sim = batched_cosine_similarity(
                image_hiddens.cuda(), caption_hiddens.cuda()
            )  # (n_images, n_captions, n_layers)
            cosine_sim = cosine_sim.cpu().float()
            pickle.dump(cosine_sim, open(cache_file, "wb"))
        model_to_cosine_sim[model_name] = cosine_sim

    for model_name, cosine_sim in model_to_cosine_sim.items():
        sims_adjusted = baseline_adjusted_cosine_similarity(cosine_sim)
        print(model_name, sims_adjusted.mean(1))


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
