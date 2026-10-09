"""STAND-IN model (NOT the team's spaCy model): a linear token classifier with context-window features.
Only used to build/test the audit pipeline when spaCy is unavailable. Writes predictions in the same
format as predict_spacy.py so audit.py works unchanged."""
import json, time
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier

L = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def feats(tokens, i):
    w = tokens[i]; lw = w.lower()
    f = {"b": 1, "w": lw, "suf3": lw[-3:], "pre3": lw[:3], "title": w.istitle(), "up": w.isupper(), "dig": w.isdigit()}
    for d in (-2, -1, 1, 2):
        j = i + d
        f[f"w{d}"] = tokens[j].lower() if 0 <= j < len(tokens) else "<pad>"
    f["w-1|w"] = f["w-1"] + "|" + lw
    f["w|w+1"] = lw + "|" + f["w1"]
    return f


def build(recs):
    X, y = [], []
    for r in recs:
        for i in range(len(r["tokens"])):
            X.append(feats(r["tokens"], i)); y.append(r["labels"][i])
    return X, y


if __name__ == "__main__":
    t0 = time.time()
    tr, te = L("data/train.json"), L("data/test.json")
    Xtr, ytr = build(tr)
    dv = DictVectorizer()
    clf = SGDClassifier(loss="hinge", alpha=2e-6, max_iter=15, tol=None, random_state=0)
    clf.fit(dv.fit_transform(Xtr), ytr)
    preds = []
    for r in te:
        X = dv.transform([feats(r["tokens"], i) for i in range(len(r["tokens"]))])
        tags = list(clf.predict(X))
        for i, t in enumerate(tags):             # repair illegal I- starts
            if t.startswith("I-") and (i == 0 or tags[i - 1][2:] != t[2:]):
                tags[i] = "B-" + t[2:]
        preds.append(tags)
    json.dump(preds, open("predictions_surrogate.json", "w"))
    print("done", round(time.time() - t0), "s")
