import math
import multiprocessing
import os
import pickle
import sys

from collections import Counter
from datasets import load_dataset
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.utils import masked_log_softmax

LANGUAGES_MAP = {
    "af": "Afrikaans",
    "am": "Amharic",
    "ar": "Arabic",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bangla",
    "ca": "Catalan",
    "ceb": "Cebuano",
    "co": "Corsican",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "fy": "Western Frisian",
    "ga": "Irish",
    "gd": "Scottish Gaelic",
    "gl": "Galician",
    "gu": "Gujarati",
    "ha": "Hausa",
    "haw": "Hawaiian",
    "hi": "Hindi",
    "hmn": "Hmong, Mong",
    "ht": "Haitian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "ig": "Igbo",
    "is": "Icelandic",
    "it": "Italian",
    "iw": "former Hebrew",
    "ja": "Japanese",
    "jv": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "ku": "Kurdish",
    "ky": "Kyrgyz",
    "la": "Latin",
    "lb": "Luxembourgish",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mi": "Maori",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "ny": "Nyanja",
    "pa": "Punjabi",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sd": "Sindhi",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sm": "Samoan",
    "sn": "Shona",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "st": "Southern Sotho",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "zh": "Chinese",
    "zu": "Zulu",
}
LANGUAGES = sorted(LANGUAGES_MAP.keys())


def load_language(language, num_texts):
    idx = 0
    texts = []
    while True:
        idx_str = f"{idx:05d}"
        try:
            dataset = load_dataset(
                "allenai/c4",
                data_files=f"multilingual/c4-{language}.tfrecord-{idx_str}-of-*.json.gz",
            )
        except ValueError:
            break
        texts.extend(dataset["train"]["text"])
        if len(texts) >= num_texts:
            break
        print(f"Loaded {len(texts)} texts for {language}")
        idx += 1
    assert len(texts) > 0
    return texts[:num_texts]


def compute_distribution(texts, tokenizer, num_processes=8):
    # tokenize the texts in parallel using and count the tokens as a distribution
    counts = Counter()

    with multiprocessing.Pool(num_processes) as pool:
        tokenized_texts = pool.map(tokenizer.tokenize, texts)

    # count the tokens
    for tokenized_text in tokenized_texts:
        counts.update(tokenized_text)

    # normalize and return a distribution
    total = sum(counts.values())
    distribution = {token: math.log(count) - math.log(total) for token, count in counts.items()}
    return distribution


def lang_logprobs(token_distributions, tokenizer, device):
    # assume uniform language prior probability
    vocab_size = tokenizer.vocab_size
    num_langs = len(LANGUAGES)
    log_probs = torch.full((num_langs, vocab_size), fill_value=-float("inf"), device=device)
    mask = torch.zeros((num_langs, vocab_size), dtype=torch.bool, device=device)
    for lang_idx, language in enumerate(tqdm(LANGUAGES)):
        for wordpiece, prob in token_distributions[language].items():
            wordpiece_idx = tokenizer.convert_tokens_to_ids(wordpiece)
            log_probs[lang_idx, wordpiece_idx] = prob
            mask[lang_idx, wordpiece_idx] = True
    # assert that the rows should be valid distributions
    assert torch.allclose(
        masked_log_softmax(log_probs, mask, dim=-1).exp().sum(dim=-1),
        torch.ones(len(log_probs), device=device),
        atol=1e-3,
    )
    return masked_log_softmax(log_probs, mask, dim=0), mask


def main(llama_path, language_distribution_output_file):
    tokenizer = AutoTokenizer.from_pretrained(llama_path, trust_remote_code=True)

    distributions = {}
    for language in tqdm(LANGUAGES_MAP.keys()):
        texts = load_language(language, 100_000)
        distribution = compute_distribution(texts, tokenizer)
        distributions[language] = distribution

    word_logprobs, lang_token_mask = lang_logprobs(
        distributions, tokenizer, device=torch.device("cpu")
    )
    pickle.dump((word_logprobs, lang_token_mask), open(language_distribution_output_file, "wb"))


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
