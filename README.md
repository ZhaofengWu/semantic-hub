# The Semantic Hub Hypothesis: Language Models Share Semantic Representations Across Languages and Modalities

This repo contains the code for our paper, [The Semantic Hub Hypothesis: Language Models Share Semantic Representations Across Languages and Modalities](https://arxiv.org/abs/2411.04986).

We have peformed some substantial cleaning/refactoring of the codebase for the release, plus some standardization of dependencies. We don't expect these to affect the results, at least qualitatively, but please let us know if you encounter any issues.

We separated the codebase into different directories for different data types. Within most, you will find a `similarity.py` that measures cross-data-type cosine similarity (Eq. 1 in our paper), a `logit_lens.py` the uses the logit lens analysis (Eq. 2 in our paper, for the most part), and an `intervention.py` for intervention experiments.

## Environment

```bash
conda create -n semantic-hub python=3.11 -y
conda activate semantic-hub
pip install -r requirements.txt
```

You may need to change the SpaCy cuda version in `requirements.txt` as needed.

## Multilingual

Our experiments require a Chinese-English dictionary, as well as tokenizer-specific language probability distributions conditioned on tokens. Run the following commands to prepare them. You may want to change the model to others. The `$dict_path` and `$lang_distribution_path` variables are used in the subsequent commands.
```bash
wget https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.zip
unzip cedict_1_0_ts_utf-8_mdbg.zip
dict_path=cedict_ts.u8

model=meta-llama/Meta-Llama-3-8B
lang_distribution_path=llama3_language_distrbution.pkl
python multilingual/compute_tokenizer_language_probabilities.py ${model} ${lang_distribution_path}
```

For the hidden state similarity experiment, you will also need the [GALE dataset](https://catalog.ldc.upenn.edu/LDC2016T25). You may need a LDC license for that. Set the `$gale_path` variable to the path of the dataset.

Now the main scripts:
```bash
python multilingual/similarity.py --model_path meta-llama/Meta-Llama-3-8B --data_path ${gale_path}
python multilingual/logit_lens.py --model_path meta-llama/Meta-Llama-3-8B --dict_path ${dict_path}
python multilingual/global_language_trends.py --model_path meta-llama/Meta-Llama-3-8B --language_distributions_path ${lang_distribution_path} --lang en  # `lang` could be `zh` too
# For the intervention experiment, we seprate it into multiple stages
python multilingual/intervention.py llama3 0 zh pos  # this script computes the intervened outputs and measures the sentiment. In our paper, for each argument in order, we consider both `llama3` and `llama2`, seed from `0` to `9`, languages `zh` and `es`, and intervention directions `pos` and `neg`
python multilingual/intervention_add_perplexity.py meta-llama/Meta-Llama-3.1-70B actadd_new_prompted*/*/*  # this script computes the perplexity of the intervened outputs
python multilingual/intervention_add_relevance.py intfloat/multilingual-e5-large actadd_new_prompted*/*/*  # this script computes the relevance of the intervened outputs
python multilingual/intervention_organize.py  # this script reports all results
```

## Arithmetic

```bash
python arithmetic/similarity.py
python arithmetic/pca.py meta-llama/Meta-Llama-3-8B
python arithmetic/logit_lens.py
python arithmetic/intervention.py
```

## Coding

```bash
python coding/logit_lens_and.py
python coding/logit_lens_semantic_role.py meta-llama/Llama-2-7b-hf
python coding/intervention.py
```

## Semantics

Set the `$cogs_dir` variable to the path of the [COGS dataset](https://github.com/najoungkim/COGS).

```bash
python semantics/similarity.py --model_path meta-llama/Meta-Llama-3-8B --data_path ${cogs_dir}/data/train.tsv  # add either `--filter_data` or `--filter_data --shuffle_data` for stronger baselines that corresopnd to Figure 9(b)/(c) in our paper
```

## Vision

You need to first download the [COCO dataset](https://cocodataset.org/#download). We used the `val2017` subset. Set the `$coco_dir` variable to the path of the dataset.

```bash
python vision/similarity.py ${coco_dir}
python vision/logit_lens_color.py llava-hf/llava-1.5-7b-hf
python vision/logit_lens_caption.py ${coco_dir} 1000
python vision/logit_lens_segmentation.py ${coco_dir} 1000
python vision/intervention.py facebook/chameleon-7b
```

## Audio

You need to download the VGGSound dataset first. You can do it from https://huggingface.co/datasets/Loie/VGGSound and untar all the files. Then run the following command to convert the mp4 files to wav files:
```bash
python batch_convert_mp4_to_wav.py ${vggsound_dir}
```

You also need to download checkpoints for the SALMONN model. Download `salmonn_v1.pth` from https://huggingface.co/tsinghua-ee/SALMONN/tree/main and `beats.pt` following the instructions in https://github.com/bytedance/SALMONN. Put both files under `SALMONN/checkpoint/`.

```bash
python audio/similarity.py ${vggsound_dir}/vggsound.csv ${vggsound_dir}/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video
python audio/logit_lens.py ${vggsound_dir}/vggsound.csv ${vggsound_dir}/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video
python audio/intervention.py ${vggsound_dir}/vggsound.csv ${vggsound_dir}/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video
```

## Tokenization Note

Llama-2 and Llama-3 tokenizers have a minor behavior difference: when encoding a token, Llama-2 by default adds a prefix space, but Llama-3 does not. Try `tokenizer.convert_ids_to_tokens(tokenizer("hello", add_special_tokens=False)["input_ids"])` to see the difference. However, in most of our experiments, the token with prefix space is more likely than the one without, so we manually add the prefix space for Llama-3 in many cases. But not always: for example, for the arithmetic experiment, because we format the data as, e.g., `5=3+`, the most likely surface form token is `2` without a prefix space, but the most likely intermediate layer token is ` two` with a prefix space. So we use the setting that is more likely for each case individually.

To complicate the matter more, in our code, sometimes we use the `add_prefix_space` boolean flag to describe when we should add a prefix space to a token (i.e., usually `False` for Llama-2 and `True` for Llama-3), but sometimes we mean that we should ensure a prefix token exists, where it would be `True` for both. Sorry about this!
