"""Symmetry (McNemar-Bowker) and marginal-homogeneity (Stuart-Maxwell) tests on the reference labels,
original vs English, with and without adjustment for clustering of records within respondents.

Cluster adjustment: the test statistic is the Wald form d' V^-1 d, where d holds the off-diagonal
differences (Bowker: n_ij - n_ji for the 3 pairs; Stuart-Maxwell: row minus column margins for 2 classes)
and V is the cluster-robust covariance, the sum over respondents of d_c d_c' (Durkalski et al., 2003,
generalised to 3 x 3 tables). With one record per respondent this reduces exactly to the usual tests.
A participant-cluster bootstrap (1,000 replicates, seed 42) gives intervals for each asymmetry.
Run: python cluster_symmetry_tests.py ANNOTATION_CSV
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import chi2

L = ["Negative", "Neutral", "Positive"]
canon = {"negative": "Negative", "neg": "Negative", "neutral": "Neutral", "neu": "Neutral",
         "positive": "Positive", "pos": "Positive", "po": "Positive"}
a = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False, encoding="utf-8-sig")
O = a.Human_Original_Sentiment.map(lambda s: canon[s.strip().lower()]).map(L.index).to_numpy()
E = a.Human_English_Sentiment.map(lambda s: canon[s.strip().lower()]).map(L.index).to_numpy()
pid = a.Participant_ID.to_numpy()
pairs = [(0, 1), (0, 2), (1, 2)]

def contrib(o, e):
    """Per-record contribution vectors for Bowker (3) and Stuart-Maxwell (2)."""
    bw = np.stack([((o == i) & (e == j)).astype(float) - ((o == j) & (e == i)) for i, j in pairs], 1)
    sm = np.stack([(o == k).astype(float) - (e == k) for k in (0, 1)], 1)
    return bw, sm

bw, sm = contrib(O, E)
df = pd.DataFrame(np.hstack([bw, sm]), columns=["b01", "b02", "b12", "s0", "s1"]); df["pid"] = pid
C = df.groupby("pid").sum()                     # per-respondent sums
n_resp = len(C)

def wald(dvec, rows):
    V = rows.T @ rows
    s = float(dvec @ np.linalg.solve(V, dvec)); return s

out = []
for name, cols, k in [("McNemar-Bowker symmetry", ["b01", "b02", "b12"], 3), ("Stuart-Maxwell marginal homogeneity", ["s0", "s1"], 2)]:
    d = df[cols].sum().to_numpy()
    s_ind = wald(d, df[cols].to_numpy())          # records independent (classical test)
    s_cl = wald(d, C[cols].to_numpy())            # respondents as clusters
    out.append((name, k, s_ind, chi2.sf(s_ind, k), s_cl, chi2.sf(s_cl, k)))
for name, k, si, pi, sc, pc in out:
    print(f"{name}: df={k} | independent: chi2={si:.2f}, p={pi:.4f} | clustered by respondent ({n_resp}): chi2={sc:.2f}, p={pc:.4f}")

# design effect per asymmetry and bootstrap intervals (percentage points of all records)
N = len(df); groups = df.groupby("pid")[["b01", "b02", "b12", "s0", "s1"]].sum()
rng = np.random.default_rng(42); idx = rng.integers(0, n_resp, size=(1000, n_resp))
cnt = df.groupby("pid").size().to_numpy()
for col, lab in [("b01", "Neg->Neu minus Neu->Neg"), ("b02", "Neg->Pos minus Pos->Neg"), ("b12", "Neu->Pos minus Pos->Neu"),
                 ("s0", "Negative share, original minus English"), ("s1", "Neutral share, original minus English")]:
    v = groups[col].to_numpy()[idx].sum(1) / cnt[idx].sum(1) * 100
    est = df[col].sum() / N * 100
    deff = (groups[col] ** 2).sum() / (df[col] ** 2).sum()
    print(f"{lab}: {est:+.2f} points [{np.quantile(v, .025):+.2f}, {np.quantile(v, .975):+.2f}], design effect {deff:.2f}")
