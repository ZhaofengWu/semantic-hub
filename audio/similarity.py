import os
import pickle
import random
import sys

random.seed(0)

import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.logit_lens import init_model
from lib.utils import batched_cosine_similarity
from SALMONN.utils import prepare_one_sample


def load_data(csv_file, videos_dir, n=None):
    metadata = pd.read_csv(
        csv_file, header=None, names=["video_id", "start_seconds", "label", "split"]
    )
    id2label = {row["video_id"]: row["label"] for _, row in metadata.iterrows()}
    all_labels = sorted(list(set(id2label.values())))
    wavs = [file for file in os.listdir(videos_dir) if file.endswith(".wav")]
    if n is not None:
        wavs = wavs[:n]
    return wavs, id2label, all_labels



def compute_audio_hiddens(wav_path, model, wav_processor):
    samples = prepare_one_sample(wav_path, wav_processor)
    prompt = [model.prompt_template.format("<Speech><SpeechHere></Speech>")]

    outputs, token_ids = model(samples, prompt=prompt)
    hidden_states = outputs.hidden_states

    all_hidden_states = torch.stack([h.squeeze(0) for h in hidden_states], dim=0)[1:]
    return all_hidden_states[:, -1, :].cpu()


def compute_label_hiddens(label, model):
    inputs = model.llama_tokenizer(label, return_tensors="pt").to(model.device)
    hidden_states = model.llama_model(inputs.input_ids, output_hidden_states=True).hidden_states
    return torch.stack([h.squeeze(0) for h in hidden_states], dim=0)[1:, -1, :].cpu()


CACHE_FILE = "audio_repr_cache_1000.pkl"


@torch.inference_mode()
def main(csv_file, videos_dir):
    if os.path.exists(CACHE_FILE):
        cosine_sim, audio_label_indices = pickle.load(open(CACHE_FILE, "rb"))
    else:
        wav_processor, model = init_model("SALMONN/configs/decode_config.yaml")
        wavs, id2label, all_labels = load_data(csv_file, videos_dir, n=1000)
        label_indices = {label: i for i, label in enumerate(all_labels)}

        audio_hiddens = []
        audio_label_indices = []
        for file in tqdm(wavs):
            id, _ = file.rsplit(".", 1)[0].rsplit("_", 1)
            label = id2label[id]

            wav_path = os.path.join(videos_dir, file)

            audio_hiddens.append(compute_audio_hiddens(wav_path, model, wav_processor))
            audio_label_indices.append(label_indices[label])

        label_hiddens = []
        for label in all_labels:
            label_hiddens.append(compute_label_hiddens(label, model))

        audio_hiddens = torch.stack(
            audio_hiddens, dim=0
        ).cuda()  # (n_audios, n_layers, hidden_size)
        label_hiddens = torch.stack(label_hiddens, dim=0).cuda()  # (n_labels, n_layers, hidden_size)
        cosine_sim = (
            batched_cosine_similarity(audio_hiddens, label_hiddens).cpu().float()
        )  # (n_audios, n_labels, n_layers)
        pickle.dump((cosine_sim, audio_label_indices), open(CACHE_FILE, "wb"))

    audio_label_mask = cosine_sim.new_zeros(cosine_sim.shape[:2], dtype=torch.bool)
    for i, label_index in enumerate(audio_label_indices):
        audio_label_mask[i, label_index] = True
    matchings = cosine_sim[audio_label_mask, :]  # (n_audios, n_layers)
    non_matchings = cosine_sim[~audio_label_mask, :]  # (n_audios * (n_labels - 1), n_layers)
    non_matching_mean = non_matchings.mean(0)

    matchings_adjusted = matchings - non_matching_mean.unsqueeze(0)
    print(matchings_adjusted.mean(0))


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
