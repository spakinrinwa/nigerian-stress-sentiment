"""Cluster-robust McNemar-Bowker and Stuart-Maxwell tests, original vs English labels, for each system
(same method as cluster_symmetry_tests.py; respondents as clusters).
Run: python model_symmetry_tests.py JOINED_PREDICTIONS_CSV"""
import sys, numpy as np, pandas as pd
from scipy.stats import chi2
L = ["Negative", "Neutral", "Positive"]; pairs = [(0, 1), (0, 2), (1, 2)]
j = pd.read_csv(sys.argv[1], dtype={"Expression_ID": str, "Participant_ID": str}, low_memory=False)
for (m, pm), g in j.groupby(["Model", "Prompt_Mode"]):
    o = g[g.Representation == "Original"].set_index("Expression_ID"); e = g[g.Representation == "English"].set_index("Expression_ID").loc[o.index]
    O = o.Predicted_Sentiment.map(L.index).to_numpy(); E = e.Predicted_Sentiment.map(L.index).to_numpy()
    bw = np.stack([((O == a) & (E == b)).astype(float) - ((O == b) & (E == a)) for a, b in pairs], 1)
    sm = np.stack([(O == k).astype(float) - (E == k) for k in (0, 1)], 1)
    C = pd.DataFrame(np.hstack([bw, sm])).groupby(o.Participant_ID.to_numpy()).sum().to_numpy()
    res = []
    for cols, k in [([0, 1, 2], 3), ([3, 4], 2)]:
        d = np.hstack([bw, sm])[:, cols].sum(0); V = C[:, cols].T @ C[:, cols]
        s = float(d @ np.linalg.solve(V, d)); res.append(f"chi2={s:.1f}, p={chi2.sf(s, k):.2g}")
    neg = 100 * ((E == 0).mean() - (O == 0).mean())
    print(f"{m} {pm}: Bowker {res[0]} | Stuart-Maxwell {res[1]} | Negative share change {neg:+.1f} points")
