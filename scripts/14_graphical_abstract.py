"""Graphical abstract for the CSL submission. Numbers from Tables 6 and A.2 (harmonised reference labels)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
O, E, G = "#D55E00", "#0072B2", "#7F7F7F"
fig = plt.figure(figsize=(13.28, 5.31), dpi=200)
# panel 1: design
ax = fig.add_axes([0.01, 0.06, 0.24, 0.80]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.text(0.01, 0.90, "Same expression,\ntwo forms", fontsize=13, weight="bold", va="bottom")
def box(y, title, text, c):
    ax.add_patch(FancyBboxPatch((0.02, y), 0.94, 0.25, boxstyle="round,pad=0.01,rounding_size=0.03", fc="white", ec=c, lw=2))
    ax.text(0.06, y + 0.19, title, color=c, weight="bold", fontsize=10)
    ax.text(0.06, y + 0.07, text, fontsize=11, style="italic")
box(0.52, "Original (here Yoruba)", "“omorr o ti remi o”", O)
box(0.12, "English interpretation", "“I am exhausted.”", E)
ax.annotate("", xy=(0.49, 0.38), xytext=(0.49, 0.51), arrowprops=dict(arrowstyle="-|>", color=G, lw=1.5))
ax.text(0.0, 0.0, "3,741 records · 314 Nigerian students", fontsize=10, color="#333")
# panel 2: stability
ax2 = fig.add_axes([0.36, 0.16, 0.22, 0.66])
names = ["Reference\nlabels", "XLM-R", "Qwen\ncultural", "AfriSenti", "Qwen\nzero-shot", "RoBERTa", "TextBlob"]
vals = [5.5, 31.3, 33.5, 35.7, 46.6, 51.6, 52.0]
cols = ["#333333"] + [G] * 6
ax2.barh(range(7)[::-1], vals, color=cols, height=0.62)
for i, v in zip(range(7)[::-1], vals): ax2.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)
ax2.set_yticks(range(7)[::-1], names, fontsize=9); ax2.set_xlim(0, 62)
ax2.set_xlabel("Records whose label changed (%)")
for s in ["top", "right"]: ax2.spines[s].set_visible(False)
fig.text(0.29, 0.90, "Models change labels;\nreference labels rarely do", fontsize=13, weight="bold", va="bottom")
# panel 3: accuracy
ax3 = fig.add_axes([0.72, 0.22, 0.26, 0.60])
sys_ = ["RoBERTa", "Qwen zero-shot", "Qwen cultural", "AfriSenti", "XLM-R", "TextBlob"]
o = [46.7, 51.9, 66.0, 63.6, 71.8, 38.9]; e = [90.1, 88.9, 87.5, 80.9, 81.7, 48.8]
ys = range(6)[::-1]
for y, a, b in zip(ys, o, e):
    ax3.plot([a, b], [y, y], color=G, lw=1.5, zorder=1)
ax3.scatter(o, list(ys), color=O, s=40, zorder=2, label="Original")
ax3.scatter(e, list(ys), color=E, marker="s", s=40, zorder=2, label="English interpretation")
ax3.axvline(70.0, color="#333", ls="--", lw=1); ax3.text(70.6, 5.45, "majority baseline\n(originals)", fontsize=8, color="#333", va="center")
ax3.set_yticks(list(ys), sys_, fontsize=9); ax3.set_xlim(30, 100)
ax3.set_xlabel("Accuracy against reference labels (%)")
for s in ["top", "right"]: ax3.spines[s].set_visible(False)
ax3.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, fontsize=8, frameon=False)
fig.text(0.63, 0.90, "Agreement with reference labels\nis higher on English", fontsize=13, weight="bold", va="bottom")
fig.savefig("graphical_abstract.pdf"); fig.savefig("graphical_abstract.tiff", dpi=200, pil_kwargs={"compression": "tiff_lzw"}); fig.savefig("graphical_abstract.png", dpi=200)
print("saved")
