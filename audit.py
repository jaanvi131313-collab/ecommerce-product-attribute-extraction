"""PERSON 3 - Model Evaluation + Audit  (Jaanvi)

Reads the TEST set + token-level BIO predictions from ANY model and produces:
  - overall + per-attribute precision / recall / F1 (entity-level, exact match, plus lenient)
  - subgroup comparison (short vs long text, product category, text-length bins,
    single- vs multi-word entities, seen vs unseen entity phrases) with bootstrap 95% CIs
  - error taxonomy + concrete wrong-prediction examples
  - gold-label quality check (are the test labels themselves complete?)
  - charts (PNG), CSV tables, metrics.json and a Word report (audit_report.docx)

Usage:
    python predict_spacy.py --model model --out predictions.json      # Shloka's model
    python audit.py --preds predictions.json --model-name "spaCy NER"
"""
import argparse, collections, json, os, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ATTRS = ["SLEEVE_STYLE", "NECKLINE", "LENGTH", "PATTERN"]
L = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
COL = {"p": "#4C72B0", "r": "#DD8452", "f": "#55A868"}


# ----------------------------------------------------------------------------- helpers
def bio_to_spans(tags):
    """BIO tags -> [(label, start_tok, end_tok_exclusive)]"""
    spans, cur = [], None
    for i, t in enumerate(tags):
        if t.startswith("B-") or (t.startswith("I-") and (cur is None or cur[0] != t[2:])):
            if cur: spans.append(tuple(cur))
            cur = [t[2:], i, i + 1]
        elif t.startswith("I-"):
            cur[2] = i + 1
        else:
            if cur: spans.append(tuple(cur))
            cur = None
    if cur: spans.append(tuple(cur))
    return spans


def overlap(a, b):
    return a[1] < b[2] and b[1] < a[2]


def prf(tp, npred, ngold):
    p = tp / npred if npred else 0.0
    r = tp / ngold if ngold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def phrase(rec, sp):
    return " ".join(rec["tokens"][sp[1]:sp[2]])


# ----------------------------------------------------------------------------- core scoring
def score_records(test, preds, train_vocab):
    """Per-record counts + a list of every error / every gold entity."""
    rows, errors, gold_rows = [], [], []
    for idx, (rec, ptags) in enumerate(zip(test, preds)):
        g, p = bio_to_spans(rec["labels"]), bio_to_spans(ptags)
        gs, ps = set(g), set(p)
        tp = len(gs & ps)
        # lenient: same label + overlapping span
        len_g = sum(any(x[0] == y[0] and overlap(x, y) for y in p) for x in g)
        len_p = sum(any(x[0] == y[0] and overlap(x, y) for y in g) for x in p)
        per = {a: dict(tp=0, np=0, ng=0) for a in ATTRS}
        for s in g: per[s[0]]["ng"] += 1
        for s in p: per[s[0]]["np"] += 1
        for s in gs & ps: per[s[0]]["tp"] += 1
        rows.append(dict(idx=idx, tp=tp, npred=len(p), ngold=len(g), len_g=len_g, len_p=len_p,
                         per=per, category=rec["category"], length_group=rec["length_group"],
                         n_words=rec["n_words"]))
        for s in g:
            seen = (s[0], phrase(rec, s).lower()) in train_vocab
            hit = s in ps
            gold_rows.append(dict(idx=idx, label=s[0], n_tok=s[2] - s[1], seen=seen, hit=hit,
                                  length_group=rec["length_group"]))
            if hit:
                continue
            ov = [y for y in p if overlap(s, y)]
            same = [y for y in ov if y[0] == s[0]]
            kind = "boundary_error" if same else ("wrong_label" if ov else "missed")
            errors.append(dict(idx=idx, type=kind, attribute=s[0], gold=phrase(rec, s),
                               predicted=(phrase(rec, (same or ov)[0]) + f" [{(same or ov)[0][0]}]") if ov else "",
                               length_group=rec["length_group"], category=rec["category"], text=rec["text"]))
        for s in p:
            if not any(overlap(s, y) for y in g):
                plaus = (s[0], phrase(rec, s).lower()) in train_vocab
                errors.append(dict(idx=idx, type="false_alarm", attribute=s[0], gold="",
                                   predicted=phrase(rec, s) + f" [{s[0]}]", length_group=rec["length_group"],
                                   category=rec["category"], text=rec["text"], plausible_unlabeled=plaus))
    return rows, pd.DataFrame(errors), pd.DataFrame(gold_rows)


def agg(rows, mask=None, attr=None):
    tp = npred = ngold = 0
    for i, r in enumerate(rows):
        if mask is not None and not mask[i]:
            continue
        if attr:
            tp += r["per"][attr]["tp"]; npred += r["per"][attr]["np"]; ngold += r["per"][attr]["ng"]
        else:
            tp += r["tp"]; npred += r["npred"]; ngold += r["ngold"]
    return tp, npred, ngold


def bootstrap_f1(rows, mask, n_boot=300, seed=0):
    idxs = [i for i, m in enumerate(mask) if m]
    rng = random.Random(seed)
    tp = np.array([rows[i]["tp"] for i in idxs]); npd = np.array([rows[i]["npred"] for i in idxs])
    ng = np.array([rows[i]["ngold"] for i in idxs])
    vals = []
    for _ in range(n_boot):
        s = np.array([rng.randrange(len(idxs)) for _ in idxs])
        vals.append(prf(tp[s].sum(), npd[s].sum(), ng[s].sum())[2])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def subgroup_table(rows, key, groupfn=None, min_n=30):
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        groups[groupfn(r) if groupfn else r[key]].append(i)
    out = []
    for name, ids in groups.items():
        mask = [False] * len(rows)
        for i in ids: mask[i] = True
        tp, npd, ng = agg(rows, mask)
        p, rc, f = prf(tp, npd, ng)
        lo, hi = bootstrap_f1(rows, mask)
        out.append(dict(subgroup=name, n_records=len(ids), n_entities=ng, precision=p, recall=rc, f1=f,
                        f1_ci_low=lo, f1_ci_high=hi, small_sample=len(ids) < min_n))
    return pd.DataFrame(out).sort_values("n_records", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------- gold label quality
def label_quality(test, train, min_count=25):
    """Lexicon of phrases that annotators tagged >= min_count times in TRAIN. Then look for those
    phrases in TEST text and see how often they were NOT tagged (possible missing annotation)."""
    cnt = collections.Counter()
    for r in train:
        for s in bio_to_spans(r["labels"]):
            cnt[(s[0], tuple(w.lower() for w in r["tokens"][s[1]:s[2]]))] += 1
    lex = [k for k, c in cnt.items() if c >= min_count]
    lex.sort(key=lambda k: -len(k[1]))
    stats = collections.defaultdict(lambda: [0, 0])      # length_group -> [total, untagged]
    examples = []
    for rec in test:
        low = [t.lower() for t in rec["tokens"]]
        used = set()
        for lab, ph in lex:
            n = len(ph)
            for i in range(len(low) - n + 1):
                if tuple(low[i:i + n]) == ph and not (set(range(i, i + n)) & used):
                    used |= set(range(i, i + n))
                    tagged = all(t.endswith(lab) and t != "O" for t in rec["labels"][i:i + n])
                    other = any(t != "O" for t in rec["labels"][i:i + n])
                    stats[rec["length_group"]][0] += 1
                    if not tagged and not other:
                        stats[rec["length_group"]][1] += 1
                        if len(examples) < 400:
                            examples.append(dict(label=lab, phrase=" ".join(ph), group=rec["length_group"], text=rec["text"]))
    return {g: dict(total=t, untagged=u, rate=u / t if t else 0) for g, (t, u) in stats.items()}, pd.DataFrame(examples), lex


# ----------------------------------------------------------------------------- charts
def chart_attr(df, path):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(df)); w = 0.26
    for k, (col, nm) in enumerate([("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]):
        b = ax.bar(x + (k - 1) * w, df[col], w, label=nm, color=list(COL.values())[k])
        for rect in b: ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + .01, f"{rect.get_height():.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(df["attribute"]); ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
    ax.set_title("Per-attribute performance (test set, exact entity match)"); ax.legend(ncol=3, loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def chart_groups(tables, path):
    fig, axes = plt.subplots(1, len(tables), figsize=(4.2 * len(tables), 3.6), sharey=True)
    for ax, (title, df) in zip(axes, tables):
        d = df[~df.small_sample] if (~df.small_sample).sum() >= 2 else df
        x = np.arange(len(d)); err = [d.f1 - d.f1_ci_low, d.f1_ci_high - d.f1]
        ax.bar(x, d.f1, color="#4C72B0", yerr=err, capsize=3)
        for xi, (f, n) in enumerate(zip(d.f1, d.n_records)): ax.text(xi, 0.03, f"{f:.2f}\nn={n}", ha="center", fontsize=7, color="white")
        ax.set_xticks(x); ax.set_xticklabels([str(s).replace(" & ", "\n& ") for s in d.subgroup], fontsize=7)
        ax.set_title(title, fontsize=9); ax.set_ylim(0, 1.05); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("F1 (95% bootstrap CI)"); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def chart_errors(err, per_attr_gold, path):
    order = ["missed", "boundary_error", "wrong_label", "false_alarm"]
    ct = err.groupby(["attribute", "type"]).size().unstack(fill_value=0).reindex(ATTRS).fillna(0)
    for o in order:
        if o not in ct: ct[o] = 0
    ct = ct[order]
    fig, ax = plt.subplots(figsize=(7, 3.4)); left = np.zeros(len(ct))
    for o, c in zip(order, ["#C44E52", "#DD8452", "#8172B2", "#937860"]):
        ax.barh(ct.index, ct[o], left=left, label=o.replace("_", " "), color=c); left += ct[o].values
    ax.invert_yaxis(); ax.set_xlabel("Number of errors"); ax.set_title("Error types per attribute")
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


# ----------------------------------------------------------------------------- report
def build_report(path, ctx):
    from docx import Document
    from docx.shared import Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    d = Document()
    sec = d.sections[0]; sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"): setattr(sec, m, Cm(2))
    d.styles["Normal"].font.name = "Calibri"; d.styles["Normal"].font.size = Pt(10.5)

    def shade(cell, hexcol):
        tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd")
        sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), hexcol); tcPr.append(sh)

    def table(df, cols, fmt=None, widths=None):
        t = d.add_table(rows=1, cols=len(cols)); t.style = "Table Grid"
        for i, c in enumerate(cols):
            cell = t.rows[0].cells[i]; cell.text = ""; run = cell.paragraphs[0].add_run(c); run.bold = True
            run.font.size = Pt(9); run.font.color.rgb = RGBColor(255, 255, 255); shade(cell, "2F5496")
        for _, row in df.iterrows():
            cells = t.add_row().cells
            for i, c in enumerate(cols):
                v = row[c]; txt = f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
                cells[i].text = ""; r = cells[i].paragraphs[0].add_run(txt); r.font.size = Pt(9)
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths): row.cells[i].width = Cm(w)
        d.add_paragraph()

    def bullets(items):
        for it in items: d.add_paragraph(it, style="List Bullet")

    h = d.add_heading("Model Evaluation & Audit", 0)
    p = d.add_paragraph(); r = p.add_run(f"E-Commerce Product Attribute Extraction  |  Person 3: Jaanvi  |  Model evaluated: {ctx['model_name']}"); r.italic = True
    if ctx.get("preview"):
        w = d.add_paragraph(); rr = w.add_run("PREVIEW: these numbers come from a stand-in model (linear token classifier), NOT the team's final spaCy model. "
                                              "Re-run audit.py on the real predictions to regenerate this report."); rr.bold = True; rr.font.color.rgb = RGBColor(192, 0, 0)

    d.add_heading("1. Setup", 1)
    bullets([f"Test set: {ctx['n_test']} product descriptions, {ctx['n_gold']} gold attribute spans (never used for training or model selection).",
             "Attributes: SLEEVE_STYLE, NECKLINE, LENGTH, PATTERN (BIO-tagged by token).",
             "Metric: entity-level precision / recall / F1. A prediction counts only if label AND exact span match. "
             "A lenient score (same label, overlapping span) is also reported to separate boundary mistakes from real misses."])

    d.add_heading("2. Overall results", 1)
    table(ctx["overall_df"], ["metric", "precision", "recall", "f1"], widths=[6, 3, 3, 3])
    d.add_heading("3. Attribute-wise results", 1)
    table(ctx["attr_df"], ["attribute", "support", "precision", "recall", "f1"], widths=[4, 2.5, 3, 3, 3])
    d.add_picture(ctx["fig_attr"], width=Cm(15))

    d.add_heading("4. Subgroup comparison", 1)
    d.add_paragraph("Short vs long text (the dataset's own length_group; short = up to 13 words, long = 14+ words):")
    table(ctx["len_df"], ["subgroup", "n_records", "n_entities", "precision", "recall", "f1", "f1_ci_low", "f1_ci_high"], widths=[2.4, 2, 2.2, 2.2, 2.2, 2, 2.2, 2.2])
    d.add_paragraph("By product category (groups with fewer than 30 test records are flagged as unreliable):")
    table(ctx["cat_df"], ["subgroup", "n_records", "n_entities", "precision", "recall", "f1", "small_sample"], widths=[5, 2, 2.2, 2.2, 2.2, 2, 2.2])
    d.add_picture(ctx["fig_groups"], width=Cm(16.5))
    d.add_paragraph("Entity-level recall by entity property:")
    table(ctx["ent_df"], ["property", "group", "n_entities", "recall"], widths=[5, 4, 3, 3])

    d.add_heading("5. Where the model fails: error analysis", 1)
    table(ctx["err_summary"], ["error type", "count", "meaning"], widths=[3.5, 2, 11])
    d.add_picture(ctx["fig_err"], width=Cm(14))
    d.add_paragraph("Examples of incorrect predictions:")
    ex = ctx["examples"]
    table(ex, ["type", "attribute", "gold", "predicted", "text"], widths=[2.3, 2.6, 2.6, 3.2, 6.3])

    d.add_heading("6. Audit findings", 1)
    bullets(ctx["findings"])
    d.add_heading("7. Model limitations", 1)
    bullets(ctx["limitations"])
    d.save(path)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--test", default="data/test.json")
    ap.add_argument("--train", default="data/train.json")
    ap.add_argument("--out", default="audit_outputs")
    ap.add_argument("--model-name", default="spaCy NER")
    ap.add_argument("--preview", action="store_true", help="mark report as produced by a stand-in model")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    test, train = L(a.test), L(a.train)
    preds = json.load(open(a.preds))
    assert len(preds) == len(test), "predictions and test set have different lengths"
    for r, p in zip(test, preds):
        assert len(r["tokens"]) == len(p), "token count mismatch - predictions must follow the dataset's tokens"

    train_vocab = set()
    for r in train:
        for s in bio_to_spans(r["labels"]):
            train_vocab.add((s[0], phrase(r, s).lower()))

    rows, err, gold = score_records(test, preds, train_vocab)
    N = len(rows)

    # ---- 1/2 overall + per attribute
    tp, npd, ng = agg(rows); P, R, F = prf(tp, npd, ng)
    lp = sum(r["len_p"] for r in rows) / npd if npd else 0
    lr = sum(r["len_g"] for r in rows) / ng if ng else 0
    lf = 2 * lp * lr / (lp + lr) if lp + lr else 0
    attr_rows = []
    for at in ATTRS:
        t, n_p, n_g = agg(rows, attr=at); p, r, f = prf(t, n_p, n_g)
        attr_rows.append(dict(attribute=at, support=n_g, precision=p, recall=r, f1=f))
    attr_df = pd.DataFrame(attr_rows)
    macro = attr_df[["precision", "recall", "f1"]].mean()
    overall_df = pd.DataFrame([
        dict(metric="Micro-average (strict)", precision=P, recall=R, f1=F),
        dict(metric="Macro-average over attributes", precision=macro.precision, recall=macro.recall, f1=macro.f1),
        dict(metric="Lenient (overlap, same label)", precision=lp, recall=lr, f1=lf)])

    # ---- 4 subgroups
    len_df = subgroup_table(rows, "length_group")
    cat_all = subgroup_table(rows, "category")
    big = {"Shirts & Tops", "Dresses"}
    cat_df = cat_all.copy()
    bins = lambda r: "1-5 words" if r["n_words"] <= 5 else "6-13 words" if r["n_words"] <= 13 else "14-40 words" if r["n_words"] <= 40 else "41+ words"
    bin_df = subgroup_table(rows, None, bins)
    bin_df["_o"] = bin_df.subgroup.map({"1-5 words": 0, "6-13 words": 1, "14-40 words": 2, "41+ words": 3}); bin_df = bin_df.sort_values("_o").drop(columns="_o")

    ent_rows = []
    for nm, g in [("1 word", gold[gold.n_tok == 1]), ("2+ words", gold[gold.n_tok >= 2])]:
        ent_rows.append(dict(property="Entity length", group=nm, n_entities=len(g), recall=g.hit.mean() if len(g) else 0))
    for nm, g in [("seen in training", gold[gold.seen]), ("unseen in training", gold[~gold.seen])]:
        ent_rows.append(dict(property="Entity phrase", group=nm, n_entities=len(g), recall=g.hit.mean() if len(g) else 0))
    ent_df = pd.DataFrame(ent_rows)

    # ---- 5 errors
    if err.empty:
        err = pd.DataFrame(columns=["idx", "type", "attribute", "gold", "predicted", "text", "length_group", "category", "plausible_unlabeled"])
    err["plausible_unlabeled"] = err.get("plausible_unlabeled", False)
    counts = err["type"].value_counts()
    meaning = {"missed": "gold attribute not predicted at all", "boundary_error": "right label but span too short/long",
               "wrong_label": "span found but given the wrong attribute", "false_alarm": "predicted attribute with no gold span nearby"}
    err_summary = pd.DataFrame([dict(**{"error type": k.replace("_", " ")}, count=int(counts.get(k, 0)), meaning=v) for k, v in meaning.items()])
    fa = err[err["type"] == "false_alarm"]
    fa_plaus = float(fa.plausible_unlabeled.astype(bool).mean()) if len(fa) else 0.0

    # diverse examples: 2 per error type, preferring different attributes
    ex = []
    for k in meaning:
        sub = err[err["type"] == k].copy()
        sub["tl"] = sub.text.str.len(); sub = sub[sub.tl < 150]
        seen_attr = set()
        for _, r in sub.sample(frac=1, random_state=3).iterrows():
            if r.attribute not in seen_attr:
                ex.append(dict(type=k.replace("_", " "), attribute=r.attribute, gold=r.gold or "-", predicted=r.predicted or "-", text=r.text)); seen_attr.add(r.attribute)
            if len(seen_attr) == 2: break
    ex_df = pd.DataFrame(ex)

    # ---- label quality
    lq, lq_ex, lex = label_quality(test, train)

    # ---- charts
    figs = {k: os.path.join(a.out, f"fig_{k}.png") for k in ("attr", "groups", "err")}
    chart_attr(attr_df, figs["attr"])
    chart_groups([("Short vs long text", len_df), ("Product category", cat_df), ("Description length", bin_df)], figs["groups"])
    chart_errors(err, gold, figs["err"])

    # ---- findings (data-driven wording)
    f_of = lambda df, n: float(df.loc[df.subgroup == n, "f1"].iloc[0]) if (df.subgroup == n).any() else float("nan")
    ci_of = lambda df, n: tuple(df.loc[df.subgroup == n, ["f1_ci_low", "f1_ci_high"]].iloc[0])
    fs, fl = f_of(len_df, "short"), f_of(len_df, "long")
    (sl, sh), (ll, lh) = ci_of(len_df, "short"), ci_of(len_df, "long")
    sep = "the 95% confidence intervals do not overlap, so the gap is real" if (sl > lh or ll > sh) else "the confidence intervals overlap, so this gap could be noise"
    worst = attr_df.sort_values("f1").iloc[0]; best = attr_df.sort_values("f1").iloc[-1]
    seen_r = ent_df[ent_df.group == "seen in training"].recall.iloc[0]; unseen_r = ent_df[ent_df.group == "unseen in training"].recall.iloc[0]
    n_unseen = int(ent_df[ent_df.group == "unseen in training"].n_entities.iloc[0])
    multi_r = ent_df[ent_df.group == "2+ words"].recall.iloc[0]; single_r = ent_df[ent_df.group == "1 word"].recall.iloc[0]
    small_cats = cat_all[cat_all.small_sample]
    top_err = counts.idxmax().replace("_", " ") if len(counts) else "n/a"
    lq_s, lq_l = lq.get("short", {}).get("rate", 0), lq.get("long", {}).get("rate", 0)

    findings = [
        f"Short vs long text: F1 is {fs:.3f} on short descriptions vs {fl:.3f} on long ones (a {abs(fs - fl) * 100:.1f}-point gap; {sep}). "
        f"Long descriptions contain far more words around each attribute, so there are more chances to miss or hallucinate one.",
        f"Weakest attribute is {worst.attribute} (F1 {worst.f1:.3f}); strongest is {best.attribute} (F1 {best.f1:.3f}).",
        (f"Generalisation: only {n_unseen} of {ng} test spans ({n_unseen / ng * 100:.1f}%) use a phrase never seen as an attribute in training, so the test set barely "
         f"tests generalisation at all (vocabulary is almost closed). Recall is {seen_r:.3f} on seen phrases vs {unseen_r:.3f} on the {n_unseen} unseen ones; "
         f"with so few unseen spans that number is indicative only, but it suggests the model relies on remembered vocabulary."),
        f"Multi-word spans are {'harder' if multi_r < single_r else 'not harder'}: recall {multi_r:.3f} for 2+ word spans vs {single_r:.3f} for single words.",
        f"The most frequent error type is '{top_err}'.",
        f"Label-quality audit: of {sum(v['total'] for v in lq.values())} test-text occurrences of phrases that annotators tag in training (e.g. 'long sleeve', 'floral'), "
        f"{lq_s * 100:.1f}% are left untagged in short descriptions and {lq_l * 100:.1f}% in long descriptions. "
        f"Gold labels are therefore incomplete, especially in long text. {fa_plaus * 100:.0f}% of the model's false alarms are phrases that annotators do tag elsewhere in training "
        f"(manual spot-checks found many were genuine mentions, e.g. 'Henley', or a list of available sleeve options). Reported precision is therefore a pessimistic estimate, "
        f"and the long-text score is the one most distorted. Boundary errors also reveal inconsistent annotation (e.g. gold 'maxi dress' in one record, 'maxi' in another).",
    ]
    if len(small_cats):
        findings.append("Some categories have very few test records (" + ", ".join(f"{r.subgroup}: n={r.n_records}" for r in small_cats.itertuples()) +
                        "), so their scores are unreliable; the model has not been properly audited on these product types.")

    limitations = [
        "Closed attribute set: only SLEEVE_STYLE, NECKLINE, LENGTH and PATTERN are extracted. Brand, colour, size and material from the original project idea are not supported.",
        "Dataset is dominated by Shirts & Tops and Dresses; other categories (shorts, coats, baby dresses) only carry PATTERN labels and are barely represented, so performance on them is not trustworthy.",
        f"Vocabulary dependence: recall is {seen_r:.2f} on phrases seen in training but {unseen_r:.2f} on the few unseen ones (n={n_unseen}), so new styles, spellings or slang are likely to be missed; the test set is too small on this to be conclusive.",
        "Noisy / incomplete / inconsistent gold labels (see audit finding) cap the achievable scores and make precision look worse than it is; some attribute words (e.g. 'polo', 'gown', 'pencil') are labelled as NECKLINE/LENGTH by convention rather than by true meaning.",
        "Every record in the dataset contains at least one attribute, so the model is never trained on descriptions with none and may over-predict on unrelated text.",
        "Exact-match scoring penalises small boundary differences (e.g. 'long sleeve' vs 'long sleeves'); the lenient score shows how much of the gap this explains.",
    ]

    ctx = dict(model_name=a.model_name, preview=a.preview, n_test=N, n_gold=ng, overall_df=overall_df, attr_df=attr_df,
               len_df=len_df, cat_df=cat_df, ent_df=ent_df, err_summary=err_summary, examples=ex_df,
               findings=findings, limitations=limitations, fig_attr=figs["attr"], fig_groups=figs["groups"], fig_err=figs["err"])
    build_report(os.path.join(a.out, "audit_report.docx"), ctx)

    # ---- save tables
    overall_df.to_csv(f"{a.out}/overall_results.csv", index=False)
    attr_df.to_csv(f"{a.out}/attribute_results.csv", index=False)
    pd.concat([len_df.assign(group_type="length_group"), cat_all.assign(group_type="category"),
               bin_df.assign(group_type="length_bin")]).to_csv(f"{a.out}/subgroup_results.csv", index=False)
    ent_df.to_csv(f"{a.out}/entity_property_results.csv", index=False)
    err.drop(columns=["idx"]).to_csv(f"{a.out}/errors.csv", index=False)
    lq_ex.to_csv(f"{a.out}/possible_missing_gold_labels.csv", index=False)
    json.dump(dict(model=a.model_name, overall=overall_df.to_dict("records"), per_attribute=attr_df.to_dict("records"),
                   subgroups=len_df.to_dict("records"), label_quality=lq, findings=findings, limitations=limitations),
              open(f"{a.out}/metrics.json", "w"), indent=2, default=float)

    pd.set_option("display.width", 200)
    print("\n=== OVERALL ==="); print(overall_df.round(3).to_string(index=False))
    print("\n=== PER ATTRIBUTE ==="); print(attr_df.round(3).to_string(index=False))
    print("\n=== SHORT vs LONG ==="); print(len_df.round(3).to_string(index=False))
    print("\n=== CATEGORY ==="); print(cat_all.round(3).to_string(index=False))
    print("\n=== LENGTH BINS ==="); print(bin_df.round(3).to_string(index=False))
    print("\n=== ENTITY PROPERTIES ==="); print(ent_df.round(3).to_string(index=False))
    print("\n=== ERRORS ==="); print(err_summary[["error type", "count"]].to_string(index=False))
    print("\nplausible-unlabeled share of false alarms:", round(fa_plaus, 3)); print("label quality:", lq)
    print("\n=== FINDINGS ==="); [print("-", f) for f in findings]
    print("\nSaved everything in", a.out)


if __name__ == "__main__":
    main()
