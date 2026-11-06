"""Figures 2-6 for 'Same meaning, different labels' (Computer Speech & Language).

Input : experiment3_results/joined_human_model_predictions.csv (written by run_experiment3_F5.py)
Output: fig2_distributions.png/.pdf ... fig6_fidelity.png/.pdf in ./figures
Run   : python make_figures.py [path/to/joined_human_model_predictions.csv]

Colour-blind-safe Okabe-Ito palette; sized for Elsevier single (90 mm) and double (190 mm) columns.
Intervals: participant-cluster percentile bootstrap, 1,000 replicates, seed 42 (as in the paper).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiment3_results/joined_human_model_predictions.csv")
OUT = Path("figures"); OUT.mkdir(exist_ok=True)
L = ["Negative", "Neutral", "Positive"]
REPS, SEED = 1000, 42
MM = 1 / 25.4
SINGLE, DOUBLE = 90 * MM, 190 * MM
OKABE = {"Negative": "#D55E00", "Neutral": "#999999", "Positive": "#0072B2",
         "Original": "#E69F00", "English": "#0072B2", "ref": "#000000"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 600, "font.family": "DejaVu Sans"})

SYSTEMS = [  # (label, Model, Prompt_Mode)
    ("RoBERTa", "Cardiff_English", "Classifier"),
    ("XLM-R", "Cardiff_Multilingual", "Classifier"),
    ("AfriSenti", "AfriSenti_AfroXLMR", "Classifier"),
    ("Qwen zero-shot", "Qwen_Qwen2_5-3B-Instruct", "ZeroShot"),
    ("Qwen cultural", "Qwen_Qwen2_5-3B-Instruct", "Cultural"),
    ("TextBlob", "TextBlob_Lexicon", "Lexicon"),
]
STUDENTS = [("MNB", "TFIDF_MultinomialNB"), ("SVM", "TFIDF_LinearSVM"), ("MLP", "TFIDF_MLP")]

j = pd.read_csv(SRC, dtype={"Expression_ID": str, "Participant_ID": str}, low_memory=False)
j["Correct"] = j["Human_Label"].eq(j["Predicted_Sentiment"])


def cond(model, mode, rep):
    return j[(j.Model == model) & (j.Prompt_Mode == mode) & (j.Representation == rep)]


def cluster_ci(df, col):
    g = df.groupby("Participant_ID")[col].agg(["sum", "count"])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(g), size=(REPS, len(g)))
    v = g["sum"].to_numpy()[idx].sum(1) / g["count"].to_numpy()[idx].sum(1)
    return np.quantile(v, [0.025, 0.975])


def macro_f1_ci(df):
    pids = df.Participant_ID.unique(); pos = {p: i for i, p in enumerate(pids)}; li = {l: i for i, l in enumerate(L)}
    cm = np.zeros((len(pids), 3, 3))
    for p, t, y in zip(df.Participant_ID, df.Human_Label, df.Predicted_Sentiment):
        cm[pos[p], li[t], li[y]] += 1

    def f1(c):
        tp = np.diag(c); P = np.divide(tp, c.sum(0), out=np.zeros(3), where=c.sum(0) > 0)
        R = np.divide(tp, c.sum(1), out=np.zeros(3), where=c.sum(1) > 0)
        return np.divide(2 * P * R, P + R, out=np.zeros(3), where=(P + R) > 0).mean()

    rng = np.random.default_rng(SEED)
    boots = [f1(cm[rng.integers(0, len(pids), len(pids))].sum(0)) for _ in range(REPS)]
    return f1(cm.sum(0)), *np.quantile(boots, [0.025, 0.975])


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight"); fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---- Fig. 2: stacked class distributions, reference labels first --------------------------------
rows = []
ref = j[(j.Model == "Cardiff_English") & (j.Prompt_Mode == "Classifier")]
for rep in ["Original", "English"]:
    s = ref[ref.Representation == rep].Human_Label.value_counts(normalize=True)
    rows.append(("Reference", rep, *[100 * s.get(l, 0) for l in L]))
for lab, m, pm in SYSTEMS:
    for rep in ["Original", "English"]:
        s = cond(m, pm, rep).Predicted_Sentiment.value_counts(normalize=True)
        rows.append((lab, rep, *[100 * s.get(l, 0) for l in L]))
d = pd.DataFrame(rows, columns=["System", "Rep", *L])
fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.6), sharey=True)
for ax, rep in zip(axes, ["Original", "English"]):
    sub = d[d.Rep == rep].iloc[::-1]; left = np.zeros(len(sub))
    for l in L:
        ax.barh(sub.System, sub[l], left=left, color=OKABE[l], label=l, edgecolor="white", linewidth=0.4)
        for y, (lft, w) in enumerate(zip(left, sub[l])):
            if w >= 8: ax.text(lft + w / 2, y, f"{w:.0f}", ha="center", va="center", color="white", fontsize=6.5)
        left += sub[l].to_numpy()
    ax.set_xlim(0, 100); ax.set_xlabel("Share of labels (%)"); ax.set_title(f"{rep} expression" if rep == "Original" else "English interpretation")
axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(1.05, 1.22), frameon=False)
save(fig, "fig2_distributions")

# ---- Fig. 3: within-system shift rate with CI ----------------------------------------------------
pts = []
for lab, m, pm in SYSTEMS:
    o = cond(m, pm, "Original")[["Expression_ID", "Participant_ID", "Predicted_Sentiment"]]
    e = cond(m, pm, "English")[["Expression_ID", "Predicted_Sentiment"]]
    x = o.merge(e, on="Expression_ID", suffixes=("_O", "_E")); x["s"] = x.Predicted_Sentiment_O.ne(x.Predicted_Sentiment_E).astype(int)
    lo, hi = cluster_ci(x, "s"); pts.append((lab, 100 * x.s.mean(), 100 * lo, 100 * hi, len(x)))
r = ref[["Expression_ID", "Participant_ID"]].merge(
    j[(j.Model == "Cardiff_English") & (j.Representation == "English")][["Expression_ID", "Human_Label"]].rename(columns={"Human_Label": "HE"}), on="Expression_ID")
r = r.merge(ref[ref.Representation == "Original"][["Expression_ID", "Human_Label"]].rename(columns={"Human_Label": "HO"}), on="Expression_ID").drop_duplicates("Expression_ID")
r["s"] = r.HO.ne(r.HE).astype(int); rlo, rhi = cluster_ci(r, "s")
pts = sorted(pts, key=lambda t: t[1])
fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
for y, (lab, v, lo, hi, n) in enumerate(pts):
    ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="o", color="#0072B2", ms=4, capsize=2, lw=1)
    ax.text(hi + 1.5, y, f"{v:.1f}", va="center", fontsize=6.5)
ax.axvspan(100 * rlo, 100 * rhi, color="#999999", alpha=0.25, lw=0)
ax.axvline(100 * r.s.mean(), color="#000000", lw=0.8, ls="--")
ax.text(100 * r.s.mean() + 1.8, 0.5, "Reference\nlabels", fontsize=6.5, va="center")
ax.set_yticks(range(len(pts)), [p[0] for p in pts]); ax.set_xlim(0, 60)
ax.set_xlabel("Records whose label changed (%)")
save(fig, "fig3_shift_rates")

# ---- Fig. 4: row-normalised confusion matrices --------------------------------------------------
sysf = SYSTEMS[:5]
fig, axes = plt.subplots(2, len(sysf), figsize=(DOUBLE, 3.6), gridspec_kw={"wspace": 0.08, "hspace": 0.35})
for c, (lab, m, pm) in enumerate(sysf):
    for rr, rep in enumerate(["Original", "English"]):
        df = cond(m, pm, rep)
        cm = pd.crosstab(df.Human_Label, df.Predicted_Sentiment).reindex(index=L, columns=L, fill_value=0).to_numpy()
        cmn = cm / cm.sum(1, keepdims=True); ax = axes[rr, c]
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for (i, k), v in np.ndenumerate(cmn):
            ax.text(k, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v > 0.6 else "black")
        ax.set_xticks(range(3), ["Neg", "Neu", "Pos"]); ax.set_yticks(range(3), ["Neg", "Neu", "Pos"] if c == 0 else [])
        if rr == 0: ax.set_title(lab)
        if c == 0: ax.set_ylabel(f"{'Original' if rep == 'Original' else 'English'}\nreference")
        if rr == 1: ax.set_xlabel("Predicted")
        ax.spines[:].set_visible(False)
save(fig, "fig4_confusion")

# ---- Fig. 5: macro-F1, original vs English, with CIs ---------------------------------------------
vals = []
for lab, m, pm in SYSTEMS:
    vals.append((lab, *macro_f1_ci(cond(m, pm, "Original")), *macro_f1_ci(cond(m, pm, "English"))))
vals = sorted(vals, key=lambda t: t[4])
fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
for y, (lab, fo, lo_o, hi_o, fe, lo_e, hi_e) in enumerate(vals):
    ax.plot([fo, fe], [y, y], color="#999999", lw=1, zorder=1)
    ax.errorbar(fo, y, xerr=[[fo - lo_o], [hi_o - fo]], fmt="o", color=OKABE["Original"], ms=4, capsize=2, lw=1, zorder=2, label="Original" if y == 0 else None)
    ax.errorbar(fe, y, xerr=[[fe - lo_e], [hi_e - fe]], fmt="s", color=OKABE["English"], ms=4, capsize=2, lw=1, zorder=2, label="English interpretation" if y == 0 else None)
ax.set_yticks(range(len(vals)), [v[0] for v in vals]); ax.set_xlim(0.2, 0.9)
ax.set_xlabel("Macro-F1 against reference labels")
ax.legend(loc="lower right", frameon=False)
save(fig, "fig5_macro_f1")

# ---- Fig. 5 (main text): forest plot of the matched English-minus-original difference --------------
# Values computed by recompute.py (all records), stageB_extra.py (translation-type) and stageC.py
# (fixed reference target; unique pairs). Percentage points with 95% participant-cluster bootstrap CIs.
import json, re
_T = json.load(open("tables.json"))
_num = lambda t: tuple(float(x.replace("\u2212", "-")) for x in re.findall(r"[\u2212+-]?\d+\.\d", t))
_a4 = {r[0]: r for r in _T["Table A.4"][1:]}; _a5 = {r[0]: r for r in _T["Table A.5"][1:]}
FOREST = {s: [_num(_a4[s][3]), _num(_a4[s][6]), _num(_a5[s][2]), _num(_a5[s][4])] for s in _T["A4_order"]}  # from Tables A.4-A.5
CHECKS = [("All records", "#000000", "o"), ("Translation-type only", "#0072B2", "s"),
          ("Fixed reference target", "#E69F00", "D"), ("Unique pairs", "#009E73", "^")]
fig, ax = plt.subplots(figsize=(SINGLE * 1.25, 3.3))
systems = list(FOREST)[::-1]
for y, s in enumerate(systems):
    for k, (lab, col, mk) in enumerate(CHECKS):
        est, lo, hi = FOREST[s][k]; yy = y + (1.5 - k) * 0.17
        ax.errorbar(est, yy, xerr=[[est - lo], [hi - est]], fmt=mk, color=col, ms=3.5, capsize=1.5, lw=0.9,
                    label=lab if y == 0 else None)
ax.axvline(0, color="#999999", lw=0.8, ls="--")
ax.set_yticks(range(len(systems)), systems); ax.set_xlim(-2, 55)
ax.set_xlabel("English minus original correctness, Δₘ (percentage points)")
ax.legend(loc="lower right", frameon=False, fontsize=6.5)
save(fig, "fig5_forest_delta")
print("forest plot written")
