"""Fig. 4: confusion matrices against the reference labels, as % of each reference class with record counts."""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
SRC = sys.argv[1] if len(sys.argv) > 1 else "experiment3_results/joined_human_model_predictions.csv"
L = ["Negative", "Neutral", "Positive"]; MM = 1/25.4; DOUBLE = 190*MM
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "savefig.dpi": 600, "font.family": "DejaVu Sans"})
j = pd.read_csv(SRC, low_memory=False)
SYS = [("RoBERTa", "Cardiff_English", "Classifier"), ("XLM-R", "Cardiff_Multilingual", "Classifier"),
       ("AfriSenti", "AfriSenti_AfroXLMR", "Classifier"), ("Qwen zero-shot", "Qwen_Qwen2_5-3B-Instruct", "ZeroShot"),
       ("Qwen cultural", "Qwen_Qwen2_5-3B-Instruct", "Cultural")]
fig, axes = plt.subplots(2, 5, figsize=(DOUBLE, 112*MM), gridspec_kw={"wspace": 0.05, "hspace": 0.3})
for c, (lab, m, pm) in enumerate(SYS):
    for rr, rep in enumerate(["Original", "English"]):
        df = j[(j.Model == m) & (j.Prompt_Mode == pm) & (j.Representation == rep)]
        df = df[df.Predicted_Sentiment.isin(L)]
        cm = pd.crosstab(df.Human_Label, df.Predicted_Sentiment).reindex(index=L, columns=L, fill_value=0).to_numpy()
        pct = 100 * cm / cm.sum(1, keepdims=True); ax = axes[rr, c]
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for (i, k), v in np.ndenumerate(pct):
            col = "white" if v > 60 else "black"
            ax.text(k, i - 0.12, f"{v:.0f}%", ha="center", va="center", fontsize=7, fontweight="bold" if i == k else "normal", color=col)
            ax.text(k, i + 0.22, f"({cm[i, k]:,})", ha="center", va="center", fontsize=5.5, color=col)
        for i in range(3):
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.2))
        ax.set_xticks(range(3), ["Neg", "Neu", "Pos"])
        ax.set_yticks(range(3), ["Neg", "Neu", "Pos"] if c == 0 else [])
        ax.tick_params(length=0)
        if rr == 0: ax.set_title(lab)
        if c == 0: ax.set_ylabel(("Original" if rep == "Original" else "English") + " reference label")
        if rr == 1: ax.set_xlabel("Predicted label")
        ax.spines[:].set_visible(False)
fig.savefig("figures/fig4_confusion.png", bbox_inches="tight"); fig.savefig("figures/fig4_confusion.pdf", bbox_inches="tight")
