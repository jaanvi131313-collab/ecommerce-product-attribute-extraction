"""Shared helpers: data loading + converting your BIO-tagged JSON into spaCy training examples.
The data files are used exactly as they are - nothing in them is changed."""
import json
import spacy
from spacy.tokens import Doc
from spacy.training import Example

DATA_DIR = "data"
MODEL_DIR = "model"
OUT_DIR = "outputs"
ATTRIBUTES = ["SLEEVE_STYLE", "NECKLINE", "LENGTH", "PATTERN"]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def bio_to_char_spans(record):
    """BIO tags over your tokens -> list of (start_char, end_char, LABEL)."""
    text, tokens, tags = record["text"], record["tokens"], record["labels"]

    # locate every token in the original text
    positions, cursor = [], 0
    for tok in tokens:
        start = text.find(tok, cursor)
        if start == -1:  # should not happen with this dataset
            return []
        positions.append((start, start + len(tok)))
        cursor = start + len(tok)

    spans, cur = [], None  # cur = [start, end, label]
    for (s, e), tag in zip(positions, tags):
        if tag.startswith("B-") or (tag.startswith("I-") and (cur is None or cur[2] != tag[2:])):
            if cur:
                spans.append(tuple(cur))
            cur = [s, e, tag[2:]]
        elif tag.startswith("I-"):
            cur[1] = e
        else:  # "O"
            if cur:
                spans.append(tuple(cur))
            cur = None
    if cur:
        spans.append(tuple(cur))
    return spans


def make_example(nlp, record):
    """Build a spaCy Example. Spans are snapped to spaCy token boundaries
    (alignment_mode='expand') so entities like '3/4 Sleeve' are not silently dropped."""
    text = record["text"]
    ref = nlp.make_doc(text)
    ents = []
    for s, e, label in bio_to_char_spans(record):
        span = ref.char_span(s, e, label=label, alignment_mode="expand")
        if span is not None and not any(
            span.start < x.end and x.start < span.end for x in ents
        ):
            ents.append(span)
    ref.ents = ents
    return Example(nlp.make_doc(text), ref)


def make_examples(nlp, records):
    return [make_example(nlp, r) for r in records]


def blank_model():
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    for label in ATTRIBUTES:
        ner.add_label(label)
    return nlp
