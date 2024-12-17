import os
import pickle
import random
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arithmetic.logit_lens import create_data, first_token_id_for_string, inflect_engine
from lib.logit_lens import init_model, get_emb_and_hidden_states

random.seed(0)


CACHE_FILE = "arithmetic_pca_cache_llama3.pkl"


@torch.inference_mode()
def main(model_path):
    data = create_data()
    num_layers = 32
    if os.path.exists(CACHE_FILE):
        pca, reduced_data = pickle.load(open(CACHE_FILE, "rb"))
    else:
        tokenizer, model = init_model(model_path)
        unemb = model.get_output_embeddings().weight.cpu()
        all_embs = []
        for op, other, target, result in tqdm(data):
            if op != "+":
                continue
            text = f"{result}={other}{op}"
            emb = get_emb_and_hidden_states(text, tokenizer, model)[:, -1, :].cpu()
            scaler = StandardScaler()
            data = scaler.fit_transform(X=emb.numpy())
            all_embs.append(torch.tensor(data))

        anchors = []
        anchor_labels = []
        for num in range(10):
            num_text = inflect_engine.number_to_words(num)
            # The different prefix space treatment is because we find the more probable setting is
            # different for each setting. This is expect -- usually tokens with the prefix space is
            # more likely, but in this particular context, the surface form does not contain spaces.
            number_unemb_nospace = unemb[
                first_token_id_for_string(tokenizer, str(num), add_prefix_space=False)
            ]
            text_unemb_space = unemb[
                first_token_id_for_string(tokenizer, num_text, add_prefix_space=True)
            ]
            anchors.extend([number_unemb_nospace, text_unemb_space])  # TODO: explain
            anchor_labels.extend([str(num), num_text])
        anchors = torch.stack(anchors, dim=0)

        pca = PCA(n_components=2)
        pca.fit(anchors)
        data = torch.cat(all_embs + [anchors], dim=0)
        reduced_data = pca.transform(data.numpy())
        pickle.dump((pca, reduced_data), open(CACHE_FILE, "wb"))

    anchor_labels = []
    for num in range(10):
        num_text = inflect_engine.number_to_words(num)
        anchor_labels.extend([str(num), num_text])

    reduced_emb_data = reduced_data[:-20].reshape(
        -1, num_layers + 1, 2
    )  # (num_examples, num_layers + 1, 2)
    for reduced_ex in random.sample(list(reduced_emb_data), 100):
        # plot reduced_ex as a sequence of lines
        for j in range(num_layers):
            plt.plot(
                [reduced_ex[j, 0], reduced_ex[j + 1, 0]],
                [reduced_ex[j, 1], reduced_ex[j + 1, 1]],
                color=cm.viridis(j / num_layers),
                alpha=0.1,
                zorder=1,
            )

    anchors_reduced = reduced_data[-20:]
    for anchor, anchor_label in zip(anchors_reduced, anchor_labels, strict=True):
        plt.annotate(anchor_label, (anchor[0], anchor[1]), fontsize=13, zorder=999999999)
        # dummy transparent point to make the label appear
        plt.scatter(anchor[0], anchor[1], alpha=0)
    # plot gradient label vertically on the side
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=0, vmax=32)),
        label="Layer",
        orientation="vertical",
        ax=plt.gca(),
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.label.set_size(21)

    plt.ylim(-3, 2.3)
    plt.xlim(-2, 1.5)
    # set tick invisible
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.savefig("arithmetic_pca.pdf")


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
