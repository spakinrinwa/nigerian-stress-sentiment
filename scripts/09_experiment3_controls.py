"""Experiment III controls and paired comparisons.
A. Training-size-matched control for Experiment III: per fold, a random subset of the participant-disjoint
   training records, of the same size as the participant- and text-disjoint training set (10 seeds).
B. Qwen effective accuracy: correct valid labels / all 3,741 records (unparsed outputs count as failures).
C. Paired participant-cluster bootstrap intervals (1,000 reps, seed 42) for principal contrasts.
Run: python experiment3_controls.py ANNOTATION_CSV JOINED_PREDICTIONS_CSV EXP3_OOF_CSV
"""
import sys, re
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

ANN, JOINED, OOF = sys.argv[1:4]
LABELS = ["Negative", "Neutral", "Positive"]
canon = {"negative": "Negative", "neg": "Negative", "neutral": "Neutral", "neu": "Neutral",
         "positive": "Positive", "pos": "Positive", "po": "Positive"}
def model(name):
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True, max_features=20000,
                          min_df=2, max_df=0.98, dtype=np.float32)
    est = {"NB": MultinomialNB(alpha=1.0), "SVM": LinearSVC(C=1.0, class_weight="balanced", random_state=42),
           "MLP": MLPClassifier(hidden_layer_sizes=(64,), activation="relu", alpha=0.005, max_iter=120,
                                early_stopping=True, n_iter_no_change=8, random_state=42)}[name]
    return make_pipeline(vec, est)
norm = lambda t: re.sub(r"\s+", " ", str(t).strip().lower())

a = pd.read_csv(ANN, dtype=str, keep_default_na=False, encoding="utf-8-sig")
for r in ["Original", "English"]:
    a[f"Human_{r}_Sentiment"] = a[f"Human_{r}_Sentiment"].map(lambda s: canon[s.strip().lower()])
TEXT = {"Original": "Original_Expression", "English": "English_Interpretation"}
NO, NE = a.Original_Expression.map(norm), a.English_Interpretation.map(norm)
folds = list(GroupKFold(n_splits=5).split(a, groups=a.Participant_ID))

# ---- A. size-matched control
rows = []
for seed in range(10):
    rng = np.random.default_rng(seed)
    trains = []
    for tr, te in folds:
        keep = tr[~(NO.iloc[tr].isin(set(NO.iloc[te])).to_numpy() | NE.iloc[tr].isin(set(NE.iloc[te])).to_numpy())]
        trains.append(rng.choice(tr, size=len(keep), replace=False))
    for rep, col in TEXT.items():
        X, y = a[col], a[f"Human_{rep}_Sentiment"]
        for m in ["NB", "SVM", "MLP"]:
            yhat = np.empty(len(a), dtype=object)
            for (_, te), tr in zip(folds, trains):
                yhat[te] = model(m).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
            rows.append(dict(Seed=seed, Model=m, Representation=rep, Accuracy=100 * accuracy_score(y, yhat),
                             Macro_F1=f1_score(y, yhat, labels=LABELS, average="macro", zero_division=0),
                             Bal_Acc=100 * balanced_accuracy_score(y, yhat), Kappa=cohen_kappa_score(y, yhat, labels=LABELS)))
A = pd.DataFrame(rows)
A.to_csv("sizematched_by_seed.csv", index=False)
print("A. Size-matched control (mean [min, max] over 10 seeds)")
print(A.groupby(["Model", "Representation"]).agg(Acc=("Accuracy", "mean"), Acc_min=("Accuracy", "min"),
      Acc_max=("Accuracy", "max"), MF1=("Macro_F1", "mean"), BA=("Bal_Acc", "mean"), K=("Kappa", "mean")).round(3).to_string())

# ---- B. Qwen effective accuracy
j = pd.read_csv(JOINED, low_memory=False)
q = j[j.Model == "Qwen_Qwen2_5-3B-Instruct"]
print("\nB. Qwen effective accuracy (correct / 3,741)")
for (rep, mode), g in q.groupby(["Representation", "Prompt_Mode"]):
    print(rep, mode, "valid", len(g), "valid-acc %.1f" % (100 * g.Correct.mean()),
          "effective %.1f" % (100 * g.Correct.sum() / 3741))

# ---- C. paired contrasts, participant-cluster bootstrap
oof = pd.read_csv(OOF)
def corr_vec(df):  # Series indexed by Expression_ID
    return df.set_index("Expression_ID").Correct.astype(float)
def pretrained(model, rep, mode):
    g = j[(j.Model == model) & (j.Representation == rep) & (j.Prompt_Mode == mode)]
    return corr_vec(g)
def exp3(model, rep, regime):
    g = oof[(oof.Model == model) & (oof.Representation == rep) & (oof.Regime == regime)].copy()
    g["Correct"] = g.Reference == g.Predicted
    return corr_vec(g)
pid = a.set_index(a.Expression_ID.astype(int)).Participant_ID
def contrast(x, y, label):
    d = pd.concat([x, y], axis=1, join="inner").dropna(); d.columns = ["x", "y"]
    d["pid"] = pid.reindex(d.index).values
    diff = 100 * (d.x - d.y)
    groups = {p: diff[d.pid == p].values for p in d.pid.unique()}
    pids = np.array(list(groups)); rng = np.random.default_rng(42); bs = []
    for _ in range(1000):
        s = rng.choice(pids, size=len(pids), replace=True)
        v = np.concatenate([groups[p] for p in s]); bs.append(v.mean())
    lo, hi = np.quantile(bs, [0.025, 0.975])
    print(f"{label}: N={len(d)}  diff={diff.mean():+.1f} [{lo:+.1f}, {hi:+.1f}]")
print("\nC. Paired contrasts (percentage points, 95% participant-cluster bootstrap)")
xlmr_o = pretrained("Cardiff_Multilingual", "Original", "Classifier")
rob_e = pretrained("Cardiff_English", "English", "Classifier")
for m in ["TFIDF_LinearSVM", "TFIDF_MLP", "TFIDF_MultinomialNB"]:
    contrast(exp3(m, "Original", "Participant-disjoint"), xlmr_o, f"{m} PD Original - XLM-R Original")
    contrast(exp3(m, "Original", "Participant- and text-disjoint"), xlmr_o, f"{m} PTD Original - XLM-R Original")
    contrast(exp3(m, "English", "Participant-disjoint"), rob_e, f"{m} PD English - RoBERTa English")
contrast(pretrained("Cardiff_Multilingual", "Original", "Classifier"),
         pretrained("AfriSenti_AfroXLMR", "Original", "Classifier"), "XLM-R - AfriSenti Original")
for rep in ["Original", "English"]:
    contrast(pretrained("Qwen_Qwen2_5-3B-Instruct", rep, "Cultural"),
             pretrained("Qwen_Qwen2_5-3B-Instruct", rep, "ZeroShot"), f"Qwen cultural - zero-shot {rep} (both valid)")
