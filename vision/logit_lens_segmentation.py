import json
import os
import pickle
import random
import sys
from PIL import Image

import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.logit_lens import init_model, get_image_logprobs
from lib.modeling_llava import LlavaForConditionalGeneration
from lib.modeling_chameleon import ChameleonForConditionalGeneration


random.seed(1)


def load_data(coco_dir, n=None):
    images = {}
    images_dir = os.path.join(coco_dir, "val2017")
    for filename in tqdm(os.listdir(images_dir)):
        image_id = int(filename.split(".")[0])
        image = Image.open(os.path.join(images_dir, filename))
        image.load()
        images[image_id] = image

    panoptic_data = json.load(open(os.path.join(coco_dir, "annotations/panoptic_val2017.json")))

    categories = {}
    for category in panoptic_data["categories"]:
        name = category["name"]
        if "-" in name:
            continue
        categories[category["id"]] = name
    image_annotations = []
    for idx, annotation in enumerate(tqdm(panoptic_data["annotations"])):
        if n is not None and idx >= n:
            break
        image = images[annotation["image_id"]]
        mask = Image.open(
            os.path.join(coco_dir, "annotations/panoptic_val2017", annotation["file_name"])
        )
        mask.load()
        segment_id_to_label = {}
        for segment in annotation["segments_info"]:
            if segment["category_id"] in categories:
                segment_id_to_label[segment["id"]] = segment["category_id"]
        labels = []
        for x in range(mask.width):
            labels_row = []
            for y in range(mask.height):
                pixel = mask.getpixel((x, y))
                assert len(pixel) == 3
                pixel_id = pixel[0] + pixel[1] * 256 + pixel[2] * 256 * 256
                if pixel_id not in segment_id_to_label:
                    labels_row.append(-1)
                else:
                    labels_row.append(segment_id_to_label[pixel_id])
            labels.append(labels_row)
        image_annotations.append((image, torch.tensor(labels)))
    return image_annotations, categories


def prepare_labels(labels, processor, patch_size):
    # convert the per-pixel labels into per-patch labels
    unique_labels = labels.unique().tolist()
    if -1 in unique_labels:
        unique_labels.remove(-1)

    all_patch_labels = []
    for label in unique_labels:
        label_tensor = torch.zeros_like(labels)
        label_tensor[labels == label] = 1
        label_tensor = label_tensor.unsqueeze(-1).expand(-1, -1, 3)
        resized_label = processor.image_processor.preprocess(
            label_tensor, do_rescale=False, do_normalize=False, return_tensors="pt"
        )["pixel_values"].squeeze(
            0
        )  # [3, h, w(=h)]
        assert (resized_label[:1, :, :] == resized_label).all()
        resized_label = resized_label[0]
        assert resized_label.shape[0] == resized_label.shape[1]
        assert resized_label.shape[0] % patch_size == 0
        num_patches_per_side = resized_label.shape[0] // patch_size
        patch_labels = torch.zeros(num_patches_per_side, num_patches_per_side, dtype=torch.bool)
        for x in range(num_patches_per_side):
            for y in range(num_patches_per_side):
                patch = resized_label[
                    x * patch_size : (x + 1) * patch_size, y * patch_size : (y + 1) * patch_size
                ]
                patch_labels[x, y] = patch.sum() / (patch_size * patch_size) > 0.5
        if patch_labels.any():
            all_patch_labels.append((patch_labels, label))

    return all_patch_labels


def compute_alignment(image_logprobs, labels, categories, model, processor):
    if isinstance(model, LlavaForConditionalGeneration):
        patch_size = model.config.vision_config.patch_size
    elif isinstance(model, ChameleonForConditionalGeneration):
        patch_size = 16
    else:
        assert False
    all_patch_labels = prepare_labels(labels, processor, patch_size)
    if len(all_patch_labels) == 0:
        return None
    all_token_logprobs = []
    for patch_labels, label_id in all_patch_labels:
        label = categories[label_id]
        label_token_id = processor.tokenizer(label, add_special_tokens=False)["input_ids"][0]
        mask = patch_labels.flatten()
        token_logprobs = image_logprobs[:, mask, label_token_id]
        all_token_logprobs.append(token_logprobs.mean(1))
    return torch.stack(all_token_logprobs, dim=1).exp().mean(1).cpu()


def main(coco_dir, n=None):
    model_to_all_alignments = {}
    for model_name, model_path, cache_file in [
        ("Llava", "llava-hf/llava-1.5-7b-hf", "segmentation_cache_llava_1000.pkl"),
        ("Chameleon", "facebook/chameleon-7b", "segmentation_cache_chameleon_1000.pkl"),
    ]:
        if os.path.exists(cache_file):
            all_alignments = pickle.load(open(cache_file, "rb"))
        else:
            n = int(n) if n is not None else None
            data, categories = load_data(coco_dir, n=n)
            processor, model = init_model(model_path)

            all_alignments = {type: [] for type in ["regular", "random_categories"]}
            for image, labels in tqdm(data):
                if (labels != -1).sum() == 0:
                    continue
                image_logprobs = get_image_logprobs(image, processor, model)
                for type in all_alignments:
                    curr_categories = categories
                    if type == "random_categories":
                        # randomly select a different category for each key
                        curr_categories = {
                            key: random.choice(list(set(curr_categories.values()) - {value}))
                            for key, value in curr_categories.items()
                        }
                    alignment = compute_alignment(
                        image_logprobs, labels, curr_categories, model, processor
                    )
                    if alignment is not None:
                        all_alignments[type].append(alignment)
            all_alignments = {
                key: torch.stack(value, dim=0) for key, value in all_alignments.items()
            }
            pickle.dump(all_alignments, open(cache_file, "wb"))
        model_to_all_alignments[model_name] = all_alignments

    for model_name, all_alignments in model_to_all_alignments.items():
        alignments = all_alignments["regular"]
        alignments_baseline = all_alignments["random_categories"]
        print(model_name, "Matching Label", alignments.mean(0))
        print(model_name, "Random Label", alignments_baseline.mean(0))


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
