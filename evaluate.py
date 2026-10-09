"""Step 2 - Evaluate on the TEST set + audit (subgroups, errors, simple explainability).
Everything is written to the outputs/ folder so you can copy numbers into the model card.

Usage:  python evaluate.py
"""
import collections, json, os
import pandas as pd
import spacy
from common import *

MIN_GROUP = 30   # subgroups smaller than this are flagged as unreliable


def prf(scores):
    return {"precision": round(float(scores["ents_p"]), 3), "recall": round(float(scores["ents_r"]), 3),
            "f1": round(scores["ents_f"], 3)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    nlp = spacy.load(MODEL_DIR)
    test = load_jsonl(f"{DATA_DIR}/test.json")
    examples = make_examples(nlp, test)

    # ---------- 1. overall + per attribute ----------
    scores = nlp.evaluate(examples)
    overall = prf(scores)
    per_attr = {k: {"precision": round(v["p"], 3), "recall": round(v["r"], 3), "f1": round(v["f"], 3)}
                for k, v in scores["ents_per_type"].items()}
    print("\n=== OVERALL (test set, %d records) ===" % len(test))
    print(overall)
    print("\n=== PER ATTRIBUTE ===")
    print(pd.DataFrame(per_attr).T.to_string())

    # ---------- 2. subgroup audit ----------
    rows = []
    for key in ["category", "length_group"]:
        groups = collections.defaultdict(list)
        for rec in test:
            groups[rec[key]].append(rec)
        for name, recs in sorted(groups.items(), key=lambda x: -len(x[1])):
            s = nlp.evaluate(make_examples(nlp, recs))
            rows.append({"subgroup_type": key, "subgroup": name, "n_records": len(recs),
                         **prf(s), "small_sample_warning": len(recs) < MIN_GROUP})
    sub_df = pd.DataFrame(rows)
    sub_df.to_csv(f"{OUT_DIR}/subgroup_results.csv", index=False)
    print("\n=== SUBGROUP RESULTS ===")
    print(sub_df.to_string(index=False))

    # ---------- 3. error analysis ----------
    errors = []
    miss_counter, wrong_counter = collections.Counter(), collections.Counter()
    for rec, eg in zip(test, examples):
        gold = {(e.start_char, e.end_char, e.label_) for e in eg.reference.ents}
        pred_doc = nlp(rec["text"])
        pred = {(e.start_char, e.end_char, e.label_) for e in pred_doc.ents}
        for s, e, l in gold - pred:
            miss_counter[(l, rec["text"][s:e].lower())] += 1
            errors.append({"type": "missed", "attribute": l, "text": rec["text"][s:e], "description": rec["text"]})
        for s, e, l in pred - gold:
            wrong_counter[(l, rec["text"][s:e].lower())] += 1
            errors.append({"type": "false_alarm", "attribute": l, "text": rec["text"][s:e], "description": rec["text"]})
    pd.DataFrame(errors).to_csv(f"{OUT_DIR}/errors.csv", index=False)
    print("\n=== MOST COMMON MISSED ENTITIES ===")
    for (l, t), c in miss_counter.most_common(10):
        print(f"  {c:3d}  {l:13s} '{t}'")
    print("=== MOST COMMON FALSE ALARMS ===")
    for (l, t), c in wrong_counter.most_common(10):
        print(f"  {c:3d}  {l:13s} '{t}'")

    # ---------- 4. explainability (simple 'top features') ----------
    # What the model has actually learned to pick up: most frequent predicted phrases per attribute.
    top = collections.defaultdict(collections.Counter)
    for rec in test:
        for ent in nlp(rec["text"]).ents:
            top[ent.label_][ent.text.lower()] += 1
    print("\n=== TOP PREDICTED PHRASES PER ATTRIBUTE (explainability) ===")
    top_json = {}
    for label in ATTRIBUTES:
        top_json[label] = top[label].most_common(8)
        print(f"  {label}: {top_json[label]}")

    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump({"overall": overall, "per_attribute": per_attr, "top_phrases": top_json}, f, indent=2)
    print(f"\nSaved: metrics.json, subgroup_results.csv, errors.csv in '{OUT_DIR}/'")


if __name__ == "__main__":
    main()
