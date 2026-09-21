import json
import re
from pathlib import Path

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.utils import pad_sequences


BASE_DIR = Path(__file__).resolve().parent.parent

TOKENIZER_PATH = BASE_DIR / "tokenizer.json"
CONFIG_PATH = BASE_DIR / "inference_config.json"


words_to_replace = {
    "capítulo",
    "editorial",
    "introducción",
    "Página",
    "freud",
    "kant",
    "www.lectulandia.com",
    "página",
    "prolegomenos"
}

phrases_to_replace = [
    "obras completas",
    "sigmund freud",
    "immanuel kant",
]


def clean_text(text, words=None, frases=None):
    if words is None:
        words = words_to_replace

    if frases is None:
        frases = phrases_to_replace

    text = text.lower()

    for frase in frases:
        text = text.replace(frase.lower(), "")

    text = re.sub(r"[«»“”‘’—–…]", " ", text)
    text = re.sub(r"\bq\b", "que", text)

    for word in words:
        text = re.sub(rf"\b{re.escape(word)}\b", "", text)

    text = re.sub(r"\b\d+(?:[\.,]\d+)?\b", "", text)
    text = re.sub(r"[^a-záéíóúüñ\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    words_list = text.strip().split()

    if len(words_list) > 20:
        words_list = words_list[10:-10]

    return " ".join(words_list)


with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    tokenizer = tokenizer_from_json(f.read())


with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    inference_config = json.load(f)


max_seq_length = inference_config["max_seq_length"]
padding = inference_config["padding"]
truncating = inference_config["truncating"]


def preprocess_text(text):
    cleaned_text = clean_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned_text])

    padded_sequence = pad_sequences(
        sequence,
        maxlen=max_seq_length,
        padding=padding,
        truncating=truncating
    )

    return padded_sequence