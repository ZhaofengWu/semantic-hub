import os
import random
import pickle
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.similarity import load_data
from lib.intervention_utils_pyvene import intervened_generate_raw
from lib.logit_lens import init_model, get_unembedding
from SALMONN.utils import prepare_one_sample


CACHE_FILE = "audio_intervention_cache.pkl"
ANIMAL_LABELS = {
    "fox barking": "fox",
    "zebra braying": "zebra",
    "penguins braying": "penguins",
    "crow cawing": "crow",
    "mouse pattering": "mouse",
    "cat caterwauling": "cat",
    "bull bellowing": "bull",
    "sheep bleating": "sheep",
    "chimpanzee pant-hooting": "chimpanzee",
    "owl hooting": "owl",
    "frog croaking": "frog",
    "snake hissing": "snake",
    "cattle mooing": "cattle",
    "chinchilla barking": "chinchilla",
    "ferret dooking": "ferret",
    "parrot talking": "parrot",
    "turkey gobbling": "turkey",
    "cat meowing": "cat",
    "elephant trumpeting": "elephant",
    "pig oinking": "pig",
    "chipmunk chirping": "chipmunk",
    "whale calling": "whale",
    "dog whimpering": "dog",
    "duck quacking": "duck",
    "cow lowing": "cow",
    "goat bleating": "goat",
    "canary calling": "canary",
    "eagle screaming": "eagle",
    "bird chirping, tweeting": "bird",
    "pheasant crowing": "pheasant",
    "dog growling": "dog",
    "warbler chirping": "warbler",
    "lions roaring": "lions",
    "cat hissing": "cat",
    "cat growling": "cat",
    "francolin calling": "francolin",
    "magpie calling": "magpie",
    "chicken clucking": "chicken",
    "dog howling": "dog",
    "coyote howling": "coyote",
    "mosquito buzzing": "mosquito",
    "dog bow-wow": "dog",
    "dog baying": "dog",
    "snake rattling": "snake",
    "mouse squeaking": "mouse",
    "otter growling": "otter",
    "horse neighing": "horse",
    "elk bugling": "elk",
    "gibbon howling": "gibbon",
    "dog barking": "dog",
    "cricket chirping": "cricket",
    "goose honking": "goose",
    "lions growling": "lions",
    "dinosaurs bellowing": "dinosaurs",
    "cheetah chirrup": "cheetah",
    "chicken crowing": "chicken",
}
MAMMALS = {
    "pig",
    "otter",
    "cow",
    "chipmunk",
    "horse",
    "chimpanzee",
    "gibbon",
    "cat",
    "cattle",
    "ferret",
    "cheetah",
    "chinchilla",
    "elk",
    "mouse",
    "bull",
    "whale",
    "coyote",
    "sheep",
    "goat",
    "lions",
    "elephant",
    "zebra",
    "fox",
    "dog",
}
NON_MAMMALS = set(ANIMAL_LABELS.values()) - MAMMALS


def do_one_sample(sample, word_old, word_new, model, layer, strength):
    prompt = [
        model.prompt_template.format("<Speech><SpeechHere></Speech> Is this animal a mammal?")
    ]
    tokenizer = model.llama_tokenizer

    yes_id = tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
    no_id = tokenizer("No", add_special_tokens=False)["input_ids"][0]

    with torch.cuda.amp.autocast(dtype=torch.float16):
        output, ids = model(sample, prompt=prompt)
        output_dist = output.logits[0, -1].softmax(0)
        yes_prob = output_dist[yes_id]
        no_prob = output_dist[no_id]
        baseline_normalized_yes_prob = yes_prob / (yes_prob + no_prob)

    pos_hidden = get_unembedding(word_new, model, tokenizer)
    neg_hidden = get_unembedding(word_old, model, tokenizer)
    position_start = 8  # BOS, _US ER : _< Spe ech >
    position_end = position_start + 88
    assert (ids[0, position_start - 1 : position_end - 1] == 0).all()
    with torch.cuda.amp.autocast(dtype=torch.float16):
        intervened_output, ids = intervened_generate_raw(
            {"verbose": torch.tensor([0]), "samples": sample, "prompt": prompt},  # dummy
            pos_hidden * strength - neg_hidden * strength,
            model,
            None,
            layer,
            list(range(position_start, position_end)),
            forward_only=True,
        )
        output_dist = intervened_output.logits[0, -1].softmax(0)
        yes_prob = output_dist[yes_id]
        no_prob = output_dist[no_id]
        intervened_normalized_yes_prob = yes_prob / (yes_prob + no_prob)

    return baseline_normalized_yes_prob.cpu().item(), intervened_normalized_yes_prob.cpu().item()


@torch.inference_mode()
def main(csv_file, videos_dir):
    layer = 13
    strengths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    if os.path.exists(CACHE_FILE):
        (
            results_baseline_mammal,
            results_intervention_mammal,
            results_baseline_non_mammal,
            results_intervention_non_mammal,
        ) = pickle.load(open(CACHE_FILE, "rb"))
    else:
        model = wav_processor = None
        wavs, id2label, _ = load_data(csv_file, videos_dir)
        wavs = [
            file
            for file in wavs
            if id2label[file.rsplit(".", 1)[0].rsplit("_", 1)[0]] in ANIMAL_LABELS
        ][:1000]

        results_baseline_mammal = {}
        results_intervention_mammal = {}
        results_baseline_non_mammal = {}
        results_intervention_non_mammal = {}
        for strength in strengths:
            print(f"target {layer}, strength {strength}")
            if model is None:
                wav_processor, model = init_model("SALMONN/configs/decode_config.yaml")
            baseline_mammal = []
            baseline_non_mammal = []
            intervention_mammal = []
            intervention_non_mammal = []
            for file in tqdm(wavs):
                id, _ = file.rsplit(".", 1)[0].rsplit("_", 1)
                label = id2label[id]
                animal = ANIMAL_LABELS[label]
                wav_path = os.path.join(videos_dir, file)
                sample = prepare_one_sample(wav_path, wav_processor)
                if animal in MAMMALS:
                    word_new = random.choice(list(NON_MAMMALS))
                elif animal in NON_MAMMALS:
                    word_new = random.choice(list(MAMMALS))
                else:
                    assert False
                baseline_result, intervened_result = do_one_sample(
                    sample, animal, word_new, model, layer, strength
                )
                if animal in MAMMALS:
                    baseline_mammal.append(baseline_result)
                    intervention_mammal.append(intervened_result)
                elif animal in NON_MAMMALS:
                    baseline_non_mammal.append(baseline_result)
                    intervention_non_mammal.append(intervened_result)

            results_baseline_mammal[(layer, strength)] = baseline_mammal
            results_intervention_mammal[(layer, strength)] = intervention_mammal
            results_baseline_non_mammal[(layer, strength)] = baseline_non_mammal
            results_intervention_non_mammal[(layer, strength)] = intervention_non_mammal

        pickle.dump(
            (
                results_baseline_mammal,
                results_intervention_mammal,
                results_baseline_non_mammal,
                results_intervention_non_mammal,
            ),
            open(CACHE_FILE, "wb"),
        )

    mammals = np.array(
        [results_baseline_mammal[(layer, strengths[0])]]
        + [results_intervention_mammal[(layer, strength)] for strength in strengths]
    )
    non_mammals = np.array(
        [results_baseline_non_mammal[(layer, strengths[0])]]
        + [results_intervention_non_mammal[(layer, strength)] for strength in strengths]
    )

    print("Mammals -> Non-mammals:", mammals.mean(axis=1))
    print("Non-mammals -> Mammals:", non_mammals.mean(axis=1))


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
