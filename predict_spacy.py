"""Run Shloka's trained spaCy model on test.json and save token-level BIO predictions.
Usage (from the team's project folder, where model/ exists):
    python predict_spacy.py --model model --test data/test.json --out predictions.json
Then:  python audit.py --preds predictions.json --model-name "spaCy NER (Shloka)"
"""
import argparse, json
import spacy


def locate_tokens(text, tokens):
    """(start,end) char offsets of each dataset token inside the text (same logic as common.py)."""
    pos, cur = [], 0
    for t in tokens:
        s = text.find(t, cur)
        if s == -1:
            s = cur            # fallback, should not happen with this dataset
        pos.append((s, s + len(t)))
        cur = s + len(t)
    return pos


def ents_to_bio(record, char_ents):
    """char_ents = [(start_char, end_char, LABEL)] -> BIO tag per dataset token (token overlaps span => inside)."""
    pos = locate_tokens(record["text"], record["tokens"])
    tags = ["O"] * len(pos)
    for s, e, lab in sorted(char_ents):
        first = True
        for i, (ts, te) in enumerate(pos):
            if ts < e and s < te and tags[i] == "O":
                tags[i] = ("B-" if first else "I-") + lab
                first = False
    return tags


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model")
    ap.add_argument("--test", default="data/test.json")
    ap.add_argument("--out", default="predictions.json")
    a = ap.parse_args()
    nlp = spacy.load(a.model)
    recs = [json.loads(l) for l in open(a.test, encoding="utf-8") if l.strip()]
    preds = []
    for r, doc in zip(recs, nlp.pipe([r["text"] for r in recs], batch_size=64)):
        preds.append(ents_to_bio(r, [(e.start_char, e.end_char, e.label_) for e in doc.ents]))
    json.dump(preds, open(a.out, "w"))
    print(f"Saved {len(preds)} predictions -> {a.out}")
