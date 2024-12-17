from collections import defaultdict
import os
import pickle
import random
import sys

random.seed(0)

import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.similarity import load_data
from lib.logit_lens import init_model
from SALMONN.utils import prepare_one_sample


def compute_logprobs(wav_path, model, wav_processor):
    samples = prepare_one_sample(wav_path, wav_processor)
    prompt = [model.prompt_template.format("<Speech><SpeechHere></Speech>")]

    outputs, token_ids = model(samples, prompt=prompt)
    hidden_states = outputs.hidden_states

    all_hidden_states = torch.stack([h.squeeze(0) for h in hidden_states], dim=0)[1:]

    unembeddings = model.llama_model.get_output_embeddings().weight.detach()
    all_hidden_logprobs = []
    for hidden_states in all_hidden_states:
        # make sure they are in the same device.
        unembeddings = unembeddings.to(hidden_states)
        dist = hidden_states @ unembeddings.T
        hidden_logprobs = dist.float().log_softmax(dim=-1)
        all_hidden_logprobs.append(hidden_logprobs)

    logprobs = torch.stack(all_hidden_logprobs, dim=0).cpu().detach()
    token_ids = token_ids.cpu()
    logprobs = logprobs[:, 1:, :]
    logprobs = logprobs[:, (token_ids == 0)[0], :]
    assert logprobs.shape[1] == 88
    return logprobs


def compute_alignment(logprobs, word, model):
    token_id = model.llama_tokenizer(word, add_special_tokens=False)["input_ids"][0]
    return logprobs[:, :, token_id].mean(1)


def parse_label(label):
    sublabels = label.split(",")
    return [word.strip() for sublabel in sublabels for word in sublabel.strip().split(" ")]


CACHE_FILE = "audio_cache_1000.pkl"


@torch.inference_mode()
def main(csv_file, videos_dir):
    if os.path.exists(CACHE_FILE):
        label2alignments, label2baselines = pickle.load(open(CACHE_FILE, "rb"))
    else:
        wav_processor, model = init_model("SALMONN/configs/decode_config.yaml")
        wavs, id2label, all_labels = load_data(csv_file, videos_dir, n=1000)
        label2words = {label: parse_label(label) for label in id2label.values()}

        label2alignments = defaultdict(list)
        label2baselines = defaultdict(list)
        for file in tqdm(wavs):
            id, _ = file.rsplit(".", 1)[0].rsplit("_", 1)
            label = id2label[id]

            wav_path = os.path.join(videos_dir, file)

            logprobs = compute_logprobs(wav_path, model, wav_processor)
            alignments = []
            for word in label2words[label]:
                alignments.append(compute_alignment(logprobs, word, model))
            avg_alignment = torch.stack(alignments).mean(0)
            label2alignments[label].append(avg_alignment)

            lables_with_no_word_overlap = [
                l for l in all_labels if len(set(label2words[l]) & set(label2words[label])) == 0
            ]
            non_label = random.choice(lables_with_no_word_overlap)
            non_label_words = label2words[non_label]
            alignments = []
            for word in non_label_words:
                alignments.append(compute_alignment(logprobs, word, model))
            avg_non_label_alignment = torch.stack(alignments).mean(0)
            label2baselines[label].append(avg_non_label_alignment)

        label2alignments = {
            label: torch.stack(alignments) for label, alignments in label2alignments.items()
        }
        label2baselines = {
            label: torch.stack(alignments) for label, alignments in label2baselines.items()
        }
        pickle.dump((label2alignments, label2baselines), open(CACHE_FILE, "wb"))

    alignments = torch.cat([alignments for alignments in label2alignments.values()], dim=0).exp()
    baselines = torch.cat([baselines for baselines in label2baselines.values()], dim=0).exp()

    print("Matching Label", alignments.mean(0))
    print("Random Label", baselines.mean(0))


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
