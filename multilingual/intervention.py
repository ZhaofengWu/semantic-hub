"""
Adapted from https://colab.research.google.com/drive/1X2ZfC4y8Jx8FbkR7m-bLi8Ifrq-8MPTO
"""

import json
import numpy as np
import random
import sys


from datasets import load_dataset, Dataset
import os
from transformers import pipeline
from transformer_lens import HookedTransformer
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.intervention_utils_actadd import intervened_generate

random.seed(0)


def initialize_model(model_name):
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(model_name)
    model.eval()
    model.to("cuda")
    return model


def write_eval_output_file(outputs, output_file):
    def convert(o):
        if isinstance(o, np.float32):
            return float(o)
        raise TypeError

    with open(output_file, "w") as f:
        print(f"Saved outputs to {output_file}")
        json.dump(outputs, f, default=convert)


def sentiment(text):
    model = pipeline(
        "text-classification",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        device=0,
        return_all_scores=True,
    )
    result = model(text, truncation=True)[0]
    pos_score = [r["score"] for r in result if r["label"][:3].lower() == "pos"][0]
    neg_score = [r["score"] for r in result if r["label"][:3].lower() == "neg"][0]
    return pos_score - neg_score


def generate_text_eval(
    propmts,
    model,
    sampling_kwargs,
    act_name,
    prompt_add,
    prompt_sub,
    coeff,
    seed,
    output_file,
    output_lg,
):
    outputs = []
    for prompt in tqdm(propmts):
        assert len(prompt) >= 3

        generated_text = intervened_generate(
            prompt=prompt,
            model=model,
            act_name=act_name,
            prompt_add=prompt_add,
            prompt_sub=prompt_sub,
            coeff=coeff,
            seed=seed,
            sampling_kwargs=sampling_kwargs,
        )
        outputs.append(
            {
                "prompt": prompt,
                "content": generated_text[len(prompt) :],
                "sentiment_score": sentiment(generated_text[len(prompt) :].strip()),
            }
        )

    write_eval_output_file(outputs, output_file)


def get_steering_words(lg, direction):
    assert direction in ["pos", "neg"]
    if lg == "en":
        prompt_add, prompt_sub = "Good", "Bad"
    elif lg == "es":
        prompt_add, prompt_sub = "Bueno", "Malo"
    elif lg == "zh":
        prompt_add, prompt_sub = "好", "坏"
    else:
        raise ValueError("Language not supported")
    if direction == "neg":
        prompt_add, prompt_sub = prompt_sub, prompt_add
    return prompt_add, prompt_sub


def main(model="llama3", seed=0, output_lg="zh", direction="pos"):
    if model == "llama3":
        model_name = "meta-llama/Meta-Llama-3-8B"
    elif model == "llama2":
        model_name = "meta-llama/Llama-2-7b-hf"
    layer = 17
    coeff = 5
    intervene_lg = "en"

    sample_n = 1000
    sampling_kwargs = dict(temperature=1.0, top_p=0.3, freq_penalty=1.0)
    seed = int(seed)
    layer = int(layer)
    coeff = int(coeff)
    assert direction in ["pos", "neg"]
    if intervene_lg == "en":
        prompt_add, prompt_sub = "Good", "Bad"
    elif intervene_lg == "es":
        prompt_add, prompt_sub = "Bueno", "Malo"
    elif intervene_lg == "zh":
        prompt_add, prompt_sub = "好", "坏"
    else:
        raise ValueError("Language not supported")
    if direction == "neg":
        prompt_add, prompt_sub = prompt_sub, prompt_add

    save_subdir = f"actadd_new_prompted{output_lg}/{model}_sentiment"
    output_file = f"{save_subdir}/n={sample_n}_l={layer}_c={coeff}_seed={seed}_{prompt_add}_{prompt_sub}_outputs.jsonl"
    if os.path.exists(output_file):
        return
    if not os.path.exists(save_subdir):
        os.makedirs(save_subdir)

    model = initialize_model(model_name)

    dataset_path = f"sentiments_{output_lg}_n{sample_n}"
    if not os.path.exists(dataset_path):
        subset = {"zh": "chinese", "es": "spanish"}[output_lg]
        dataset = load_dataset("tyqiangz/multilingual-sentiments", subset, split="train")
        sampled_tox_dataset_indices = random.sample(range(len(dataset)), sample_n)
        dataset = dataset.select(sampled_tox_dataset_indices)
        dataset.save_to_disk(dataset_path)
    else:
        dataset = Dataset.load_from_disk(dataset_path)

    # We add textual prompts to maintain language consistency in generation. Otherwise the model
    # tends to code-switch into English.
    if output_lg == "zh":
        prefix = "接下来的文字全部是中文的。"
    elif output_lg == "es":
        prefix = "Todo el texto siguiente está en español. "
    else:
        assert False
    prompts = [prefix + d["text"] for d in dataset]

    generate_text_eval(
        propmts=prompts,
        model=model,
        sampling_kwargs=sampling_kwargs,
        act_name=layer,
        prompt_add=prompt_add,
        prompt_sub=prompt_sub,
        coeff=coeff,
        seed=seed,
        output_file=output_file,
        output_lg=output_lg,
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
