#!/usr/bin/env python3
"""
Regenerates every number for Table V / Table VI (C-2) and the paired test for the
sequence-baseline comparison (C-4, M-8) from per-image predictions of the FINAL run.

Input CSV(s) (one row per test image, all 212 images, header required):
    image,gt,pred
  gt   = 5-digit ground-truth string (keep leading zeros; read as text)
  pred = model output string (empty string if nothing was read)

Usage:
  python eval_final_run.py --main paddle_final.csv
  python eval_final_run.py --main paddle_final.csv --other cnn_final.csv --other-name "Segmented CNN"
  python eval_final_run.py --main paddle_final.csv --other scratch_seq.csv --other-name "From-scratch seq."

Outputs (in ./eval_out): table_v_rows.tex, confusion_matrix.pdf/.png (all 10 labels), summary.txt
"""
import argparse, csv, math, os
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIG = "0123456789"

def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["image"]: (r["gt"].strip(), (r["pred"] or "").strip()) for r in rows}

def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return 100*(c-h), 100*(c+h)

def mcnemar_exact(b, c):
    """two-sided exact binomial test on discordant pairs"""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k+1)) / 2**n
    return min(1.0, 2*p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--other"); ap.add_argument("--other-name", default="Other")
    ap.add_argument("--out", default="eval_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D = load(a.main); N = len(D)
    out = []
    # ---- end-to-end exact match (wrong length = error) ----
    ok = {i: (g == p) for i, (g, p) in D.items()}
    k = sum(ok.values()); lo, hi = wilson(k, N)
    out.append(f"End-to-end exact match: {100*k/N:.2f}% ({k}/{N})  95% Wilson CI [{lo:.2f}, {hi:.2f}]")
    wl = [i for i, (g, p) in D.items() if len(p) != len(g)]
    out.append(f"Wrong-length predictions: {len(wl)}  -> same-length subset = {N-len(wl)} images, "
               f"{5*(N-len(wl))} characters")
    # ---- character-level report on same-length subset ----
    cm = np.zeros((10, 10), int)
    for g, p in D.values():
        if len(p) == len(g):
            for x, y in zip(g, p):
                if x in DIG and y in DIG: cm[int(x), int(y)] += 1
    sup = cm.sum(1); tp = np.diag(cm); pred_n = cm.sum(0)
    prec = np.divide(tp, pred_n, out=np.zeros(10), where=pred_n > 0)
    rec = np.divide(tp, sup, out=np.zeros(10), where=sup > 0)
    f1 = np.divide(2*prec*rec, prec+rec, out=np.zeros(10), where=(prec+rec) > 0)
    tot = sup.sum(); acc = tp.sum()/tot
    rows = [f"{c} & {prec[c]:.2f} & {rec[c]:.2f} & {f1[c]:.2f} & {sup[c]} \\\\" for c in range(10)]
    T = f"1{{,}}{tot % 1000:03d}" if tot >= 1000 else str(tot)
    rows += [r"\midrule",
             f"\\textbf{{Accuracy}} & \\multicolumn{{3}}{{c}}{{{acc:.2f}}} & {T} \\\\",
             f"\\textbf{{Macro avg.}} & {prec.mean():.2f} & {rec.mean():.2f} & {f1.mean():.2f} & {T} \\\\",
             f"\\textbf{{Weighted avg.}} & {(prec*sup).sum()/tot:.2f} & {(rec*sup).sum()/tot:.2f} & {(f1*sup).sum()/tot:.2f} & {T} \\\\"]
    open(f"{a.out}/table_v_rows.tex", "w").write("\n".join(rows) + "\n")
    out.append(f"Character-level accuracy (same-length subset): {100*acc:.2f}% over support = {tot}")
    out.append("Per-class support: " + ", ".join(f"{c}:{sup[c]}" for c in range(10)))
    # ---- confusion matrix with ALL ten labels (also fixes M-6) ----
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(list(DIG)); ax.set_yticklabels(list(DIG))
    ax.set_xlabel("Predicted digit"); ax.set_ylabel("True digit")
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max()/2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04); fig.tight_layout()
    fig.savefig(f"{a.out}/confusion_matrix.pdf"); fig.savefig(f"{a.out}/confusion_matrix.png", dpi=300)
    # ---- error clustering on same-length images (C-3b) ----
    errs = {i: sum(x != y for x, y in zip(g, p)) for i, (g, p) in D.items() if len(p) == len(g)}
    n_s = len(errs); fail = [e for e in errs.values() if e > 0]
    p_char = acc
    exp_fail = n_s*(1-p_char**5)
    exp_per = 5*(1-p_char)/(1-p_char**5) if p_char < 1 else float("nan")
    obs_per = (sum(fail)/len(fail)) if fail else float("nan")
    out.append(f"Clustering (same-length images): failing={len(fail)} (indep. expects {exp_fail:.1f}); "
               f"errors/failing image observed {obs_per:.2f} vs expected {exp_per:.2f}")
    out.append("Digit-error-count histogram per image: " + str(dict(sorted(Counter(errs.values()).items()))))
    # ---- paired comparison (C-4 / M-8) ----
    if a.other:
        O = load(a.other); common = sorted(set(D) & set(O))
        if len(common) != N: out.append(f"WARNING: only {len(common)} of {N} images shared; check filenames")
        b = sum(1 for i in common if D[i][0] == D[i][1] and O[i][0] != O[i][1])  # main right, other wrong
        c = sum(1 for i in common if D[i][0] != D[i][1] and O[i][0] == O[i][1])  # main wrong, other right
        ko = sum(O[i][0] == O[i][1] for i in common); l2, h2 = wilson(ko, len(common))
        out.append(f"{a.other_name}: {100*ko/len(common):.2f}% ({ko}/{len(common)}) CI [{l2:.2f}, {h2:.2f}]")
        out.append(f"Discordant pairs: main-only correct = {b}, {a.other_name}-only correct = {c}; "
                   f"exact McNemar p = {mcnemar_exact(b, c):.3g}")
    txt = "\n".join(out); open(f"{a.out}/summary.txt", "w").write(txt + "\n"); print(txt)

if __name__ == "__main__":
    main()