# E-commerce Attribute Extraction (NER) - Run guide

1. Put train.json, val.json, test.json, labels.txt inside the `data/` folder (already there if you unzipped ca3_data.zip into it).
2. `pip install -r requirements.txt`
3. `python train.py --quick`   -> 1-2 min sanity check (2000 records)
4. `python train.py`           -> full training (best model saved to `model/`)
5. `python evaluate.py`        -> test results + subgroup audit + errors (saved to `outputs/`)
6. `python explain.py "A black floral dress with long sleeves"`  -> explainability
7. `python gui.py`             -> type a sentence, get attributes
