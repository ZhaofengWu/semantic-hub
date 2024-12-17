from collections import defaultdict
import os
import random
import pickle
import sys

from datasets import load_dataset, Dataset
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.logit_lens import init_model

random.seed(0)


def parse_instance(text, formal, ignore_start=True):
    name2role = {}
    role = None
    i = 0
    while i < len(formal):
        c = formal[i]
        if c == ".":
            i += 1
            while formal[i] == " ":
                i += 1
            # role
            end_pos = formal[i:].index(" ")
            role = formal[i : i + end_pos]
            i += end_pos
        elif c == "(" or c == ",":
            i += 1
            while formal[i] == " ":
                i += 1
            if formal[i] == "x":
                i += 1
                continue
            # proper noun
            end_pos = formal[i:].index(" ")
            name = formal[i : i + end_pos]
            i += end_pos
            assert role is not None and name2role.get(name) in {None, role}
            if not (ignore_start and text.startswith(name)):
                # we don't want names that are at the beginning of the sentence
                name2role[name] = role
            role = None
        else:
            i += 1
    return name2role


def process_data(dataset: Dataset, ignore_start=True):
    data = []
    for ex in dataset:
        if ex["type"] == "primitive":
            continue
        name2role = parse_instance(ex["text"], ex["formal"], ignore_start=ignore_start)
        data.append((ex["text"], ex["formal"], name2role))
    return data


def shuffle_names(data):
    all_names = {name for _, _, name2role in data for name in name2role.keys()}
    data_new = []
    all_texts = {text for text, _, _ in data}
    all_formals = {formal for _, formal, _ in data}
    for text, formal, name2role in tqdm(data, desc="Shuffling names"):
        if len(name2role) == 0:
            continue
        non_existing_names = list(all_names - set(name2role.keys()))
        while True:
            new_names = random.sample(non_existing_names, len(name2role))
            name_map = {
                name: new_name
                for name, new_name in zip(list(name2role.keys()), new_names, strict=True)
            }
            new_text = text
            new_formal = formal
            for old_name, new_name in name_map.items():
                new_text = new_text.replace(old_name, new_name)
                new_formal = new_formal.replace(old_name, new_name)

            if new_text in all_texts or new_formal in all_formals:
                continue
            words = new_text.split()
            pns = [word for word in words if word[0].isupper()]
            assert len(pns) == len(set(pns))
            new_name2role = {name_map[old_name]: role for old_name, role in name2role.items()}
            data_new.append((new_text, new_formal, new_name2role))
            break
    return data_new


def get_role_logprobs(tokenizer, model, sent, name, roles, add_prefix_space=True):
    inputs = tokenizer(sent, return_tensors="pt").to(model.device)
    tokens = [tokenizer.decode(i).strip() for i in inputs["input_ids"][0]]
    if name not in tokens:
        return None
    assert tokens.count(name) == 1
    name_idx = tokens.index(name)
    all_hidden_states = model(
        inputs.input_ids[:, :name_idx], output_hidden_states=True
    ).hidden_states
    all_hidden_states = torch.stack([h.squeeze(0) for h in all_hidden_states], dim=0)[1:]
    unembeddings = model.get_output_embeddings().weight.detach()
    all_hidden_logprobs = []
    for hidden_states in all_hidden_states:
        dist = hidden_states @ unembeddings.T
        hidden_logprobs = dist.float().log_softmax(dim=-1)
        all_hidden_logprobs.append(hidden_logprobs)
    logprobs = torch.stack(all_hidden_logprobs, dim=0).cpu()[:, -1, :].exp()

    role_logprobs = []
    for role in roles:
        role_vocab_idx = tokenizer(
            (" " if add_prefix_space else "") + role, add_special_tokens=False
        )["input_ids"][0]
        role_logprobs.append(logprobs[:, role_vocab_idx])
    return torch.stack(role_logprobs, dim=0)


CACHE_FILE = "cogs_semantic_role_cache_shuffled.pkl"


@torch.inference_mode()
def main(model_path, cogs_path):
    dataset = load_dataset(
        "csv",
        sep="\t",
        data_files={
            "train": os.path.join(cogs_path, "train.tsv"),
            "validation": os.path.join(cogs_path, "dev.tsv"),
            "test": os.path.join(cogs_path, "test.tsv"),
        },
        column_names=["text", "formal", "type"],
    )
    data = process_data(dataset["train"], ignore_start=False)
    data = shuffle_names(data)
    # throw away names that are in the beginning of the sentence
    new_data = []
    for sent, formal, name2role in data:
        new_name2role = {
            name: role for name, role in name2role.items() if not sent.startswith(name)
        }
        if len(new_name2role) == 0:
            continue
        new_data.append((sent, formal, new_name2role))
    data = new_data

    all_roles = sorted(list({role for _, _, name2role in data for role in name2role.values()}))

    if os.path.exists(CACHE_FILE):
        role_logprobs_matrix = pickle.load(open(CACHE_FILE, "rb"))
    else:
        tokenizer, model = init_model(model_path)

        if model_path == "meta-llama/Meta-Llama-3-8B":
            add_prefix_space = True
        elif model_path == "meta-llama/Llama-2-7b-hf":
            add_prefix_space = False
        else:
            assert False
        role_logprobs_matrix = defaultdict(list)
        for sent, formal, name2role in tqdm(data):
            for name, gold_role in name2role.items():
                role_logprobs = get_role_logprobs(
                    tokenizer, model, sent, name, all_roles, add_prefix_space=add_prefix_space
                )
                if role_logprobs is None:
                    continue
                for pred_role, logprobs in zip(all_roles, role_logprobs, strict=True):
                    role_logprobs_matrix[gold_role, pred_role].append(logprobs)
        for gold_role, pred_role in role_logprobs_matrix:
            role_logprobs_matrix[gold_role, pred_role] = torch.stack(
                role_logprobs_matrix[gold_role, pred_role], dim=0
            )
        pickle.dump(role_logprobs_matrix, open(CACHE_FILE, "wb"))

    for gold_role in all_roles:
        for pred_role in all_roles:
            print(
                f"Gold: {gold_role}, Pred: {pred_role}",
                role_logprobs_matrix[gold_role, pred_role].mean(0),
            )

    for pred_role in all_roles:
        baseline = torch.cat(
            [role_logprobs_matrix[gold_role, pred_role] for gold_role in all_roles], dim=0
        ).mean(0)
        for gold_role in all_roles:
            role_logprobs_matrix[gold_role, pred_role] -= baseline.unsqueeze(0)

    for gold_role in all_roles:
        for pred_role in all_roles:
            print(
                f"Gold: {gold_role}, Pred: {pred_role}",
                role_logprobs_matrix[gold_role, pred_role].mean(0),
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
