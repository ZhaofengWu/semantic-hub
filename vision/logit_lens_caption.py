import json
import os
import pickle
import sys
from PIL import Image

import spacy
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.logit_lens import init_model, get_image_logprobs


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


def compute_alignment(image_logprobs, keywords, processor):
    keyword_ids: list[int] = [
        processor.tokenizer(keyword, add_special_tokens=False)["input_ids"][0]
        for keyword in keywords
    ]
    keyword_ids = torch.tensor(keyword_ids).to(image_logprobs.device)
    token_logprobs = image_logprobs[:, :, keyword_ids]  # (n_layers, n_image_tokens, n_keywords)
    return token_logprobs.exp().mean(1).sum(1).cpu()


NOUN_POS = ["NOUN", "PROPN"]


def select_irrelevant_caption(spacy_caption, spacy_captions):
    def similarity(c1, c2):
        return len(set([c.text.lower() for c in c1]) & set([c.text.lower() for c in c2]))

    caption_sims = {c: similarity(spacy_caption, c) for c in spacy_captions}
    return min(caption_sims, key=caption_sims.get)


def main(coco_dir, n=10):
    model_to_all_alignments = {}
    for model_name, model_path, cache_file in [
        ("Llava", "llava-hf/llava-1.5-7b-hf", "caption_cache_llava_1000.pkl"),
        ("Chameleon", "facebook/chameleon-7b", "caption_cache_chameleon_1000.pkl"),
    ]:
        if os.path.exists(cache_file):
            all_alignments = pickle.load(open(cache_file, "rb"))
        else:
            spacy.require_gpu()
            spacy_model = spacy.load("en_core_web_trf")

            full_data = load_data(coco_dir)
            full_data = [(image, spacy_model(caption)) for image, caption in tqdm(full_data)]
            del spacy_model

            data = full_data[: int(n)]
            processor, model = init_model(model_path)

            all_alignments = {key: [] for key in ["nouns", "irrelevant_nouns"]}
            for image, spacy_caption in tqdm(data):
                irrelevant_caption = select_irrelevant_caption(
                    spacy_caption, [c for _, c in full_data]
                )
                image_logprobs = get_image_logprobs(image, processor, model)
                for key in all_alignments:
                    nouns_tokens = [token for token in spacy_caption if token.pos_ in NOUN_POS]
                    nouns = [token.text for token in nouns_tokens]
                    assert len(nouns) > 0
                    if key == "nouns":
                        keywords = [token.text for token in spacy_caption if token.pos_ in NOUN_POS]
                    elif key == "irrelevant_nouns":
                        keywords = [
                            token.text for token in irrelevant_caption if token.pos_ in NOUN_POS
                        ]
                    else:
                        assert False
                    if len(keywords) == 0:
                        print(f"No {key} found")
                        continue
                    alignment = compute_alignment(image_logprobs, keywords, processor)
                    alignment = alignment / len(keywords) * len(nouns)
                    all_alignments[key].append(alignment)
            all_alignments = {
                key: torch.stack(value, dim=0) for key, value in all_alignments.items()
            }
            pickle.dump(all_alignments, open(cache_file, "wb"))
        model_to_all_alignments[model_name] = all_alignments

    for model_name, all_alignments in model_to_all_alignments.items():
        alignments = all_alignments["nouns"]
        alignments_baseline = all_alignments["irrelevant_nouns"]
        print(model_name, "Matching caption", alignments.mean(0))
        print(model_name, "Irrelevant caption", alignments_baseline.mean(0))


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
