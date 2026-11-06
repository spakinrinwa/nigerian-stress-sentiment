"""Fig. 6 (accuracy by language) and Fig. 7 (sensitivity analyses) for the CSL manuscript.
Inputs: T10_subgroups.csv (language subgroups) and tables.json (Table A.8, from script 10)."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
OUT = Path("figures"); MM = 1/25.4; DOUBLE = 190*MM
O, E, GREY = "#E69F00", "#0072B2", "#BBBBBB"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600, "font.family": "DejaVu Sans"})
def save(fig, n):
    fig.savefig(OUT/f"{n}.png", bbox_inches="tight"); fig.savefig(OUT/f"{n}.pdf", bbox_inches="tight"); plt.close(fig)

# ---- Fig. 6 ----
s = pd.read_csv("T10_subgroups.csv"); s = s[(s.Feature == "Lang") & ~s.Condition.str.contains("TFIDF")].copy()
s[["M", "R", "P"]] = s.Condition.str.split(r" \| ", expand=True)
SYS = [("RoBERTa", "Cardiff_English", "Classifier"), ("XLM-R", "Cardiff_Multilingual", "Classifier"),
       ("AfriSenti", "AfriSenti_AfroXLMR", "Classifier"), ("Qwen zero-shot", "Qwen_Qwen2_5-3B-Instruct", "ZeroShot"),
       ("Qwen cultural", "Qwen_Qwen2_5-3B-Instruct", "Cultural")]
LANG = [("English", "English"), ("Pidgin", "Nigerian Pidgin"), ("Yoruba", "Yoruba"), ("Yoruba exclamation", "Yoruba exclamation"),
        ("Unclear", "Unclear"), ("Pidgin/English", "Pidgin/English"), ("Nigerian English", "Nigerian English"),
        ("General Nigerian usage", "General Nigerian usage"), ("Other/mixed (<25 each)", "Other or mixed")]
Nref = s[(s.M == "Cardiff_English") & (s.R == "Original")].set_index("Value").N
fig, axes = plt.subplots(1, 5, figsize=(DOUBLE, 78*MM), sharey=True)
ys = np.arange(len(LANG))[::-1]
rows = []
for ax, (lab, m, p) in zip(axes, SYS):
    for y, (v, nm) in zip(ys, LANG):
        g = s[(s.M == m) & (s.P == p) & (s.Value == v)].set_index("R").Accuracy
        go, ge = g.get("Original", np.nan), g.get("English", np.nan)
        rows.append((lab, nm, go, ge))
        if not np.isnan(ge): ax.plot([go, ge], [y, y], color=GREY, lw=1, zorder=1)
        ax.scatter(go, y, s=16, color=O, marker="o", zorder=2, label="Original" if y == ys[0] else None)
        if not np.isnan(ge): ax.scatter(ge, y, s=16, color=E, marker="s", zorder=2, label="English interpretation" if y == ys[0] else None)
    ax.set_title(lab); ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100]); ax.grid(axis="x", color="#EEEEEE", lw=0.6); ax.set_axisbelow(True)
axes[0].set_yticks(ys, [f"{nm} ({Nref[v]:,})" for v, nm in LANG])
fig.supxlabel("Accuracy against the reference labels (%)", fontsize=8, y=-0.03)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.04))
save(fig, "fig6_language")
pd.DataFrame(rows, columns=["System", "Language", "Original", "English"]).round(1).to_csv(OUT/"fig6_values.csv", index=False)

# ---- Fig. 7 (accuracy levels per system under each analysis) ----
t = json.load(open("tables.json"))["Table A.8"]
hdr, body = t[0], t[1:]
df = pd.DataFrame([[r[0]] + [float(x) for x in r[1:]] for r in body], columns=["Cond"] + hdr[1:])
labels = ["All records (3,741)", "Excl. code 999 (3,427)", "Excl. 46 corrupted (3,695)", "Excl. 64 stale English (3,677)",
          "Excl. all three (3,324)", "Unique pairs only (2,190)", "Translation-type only (3,023)", "Gloss-type only (718)"]
order = ["RoBERTa", "XLM-R", "AfriSenti", "Qwen zero-shot", "Qwen cultural", "TextBlob"]
fig, axes = plt.subplots(2, 3, figsize=(DOUBLE, 120*MM), sharex=True, sharey=True)
ys = np.arange(len(labels))[::-1]
for ax, sname in zip(axes.flat, order):
    ro = df[df.Cond == f"{sname} O"].iloc[0]; re_ = df[df.Cond == f"{sname} E"].iloc[0]
    ax.axhspan(ys[0] - 0.45, ys[0] + 0.45, color="#F2F2F2", zorder=0)
    for y, a in zip(ys, hdr[1:]):
        ax.plot([ro[a], re_[a]], [y, y], color=GREY, lw=1, zorder=1)
        ax.scatter(ro[a], y, s=16, color=O, marker="o", zorder=2, label="Original" if y == ys[0] else None)
        ax.scatter(re_[a], y, s=16, color=E, marker="s", zorder=2, label="English interpretation" if y == ys[0] else None)
    ax.set_title(sname); ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", color="#EEEEEE", lw=0.6); ax.set_axisbelow(True)
for ax in axes[:, 0]: ax.set_yticks(ys, labels)
fig.supxlabel("Accuracy against the reference labels (%)", fontsize=8, y=0.0)
h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.6, 1.03))
fig.tight_layout()
save(fig, "fig7_sensitivity")
