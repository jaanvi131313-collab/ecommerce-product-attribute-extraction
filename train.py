"""Step 1 - Train the NER model on the FULL training set.

Usage:
    python train.py                 # full data, up to 15 epochs, early stopping
    python train.py --quick         # 2000 examples, 3 epochs (just to test that everything runs)
    python train.py --epochs 8 --n-train 10000
"""
import argparse, json, os, random, time
from spacy.util import minibatch, compounding
from common import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--n-train", type=int, default=None, help="use only the first N training records")
    ap.add_argument("--patience", type=int, default=3, help="stop if val F1 doesn't improve for N epochs")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.epochs, args.n_train = 3, 2000

    random.seed(42)
    train = load_jsonl(f"{DATA_DIR}/train.json")
    val = load_jsonl(f"{DATA_DIR}/val.json")
    if args.n_train:
        train = train[: args.n_train]

    nlp = blank_model()
    train_ex = make_examples(nlp, train)
    val_ex = make_examples(nlp, val)
    print(f"Training on {len(train_ex)} examples, validating on {len(val_ex)}")

    optimizer = nlp.initialize(lambda: train_ex)
    best_f1, bad_epochs, history = -1, 0, []
    os.makedirs(OUT_DIR, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        random.shuffle(train_ex)
        losses = {}
        for batch in minibatch(train_ex, size=compounding(16.0, 64.0, 1.001)):
            nlp.update(batch, sgd=optimizer, drop=0.2, losses=losses)
        scores = nlp.evaluate(val_ex)
        f1 = scores["ents_f"]
        history.append({"epoch": epoch, "loss": round(float(losses["ner"]), 2),
                        "val_precision": round(float(scores["ents_p"]), 4),
                        "val_recall": round(float(scores["ents_r"]), 4),
                        "val_f1": round(float(f1), 4)})
        print(f"Epoch {epoch:2d} | loss {losses['ner']:9.1f} | val P {scores['ents_p']:.3f} "
              f"R {scores['ents_r']:.3f} F1 {f1:.3f} | {time.time() - t0:.0f}s")
        if f1 > best_f1:
            best_f1, bad_epochs = f1, 0
            nlp.to_disk(MODEL_DIR)          # keep the best model so far
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("No improvement - stopping early.")
                break

    with open(f"{OUT_DIR}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nBest validation F1: {best_f1:.3f}  -> model saved in '{MODEL_DIR}/'")


if __name__ == "__main__":
    main()
