"""Step 3 (optional) - Occlusion explainability for single sentences.
Removes one word at a time and checks which extracted attributes disappear.
A word whose removal kills an attribute is a word the model relied on.

Usage:  python explain.py "A black floral dress with long sleeves and a V-neck"
"""
import sys, spacy
from common import MODEL_DIR


def extract(nlp, text):
    return {(e.text.lower(), e.label_) for e in nlp(text).ents}


def occlusion(nlp, text):
    base = extract(nlp, text)
    words = text.split()
    results = []
    for i, w in enumerate(words):
        reduced = " ".join(words[:i] + words[i + 1:])
        lost = base - extract(nlp, reduced)
        results.append((w, lost))
    return base, results


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "A black floral dress with long sleeves and a V-neck"
    nlp = spacy.load(MODEL_DIR)
    base, res = occlusion(nlp, text)
    print("Sentence:", text)
    print("Extracted:", sorted(base))
    print("\nWord-removal impact:")
    for w, lost in res:
        flag = f"  -> removing it loses {sorted(lost)}" if lost else ""
        print(f"  {w:15s}{flag}")
