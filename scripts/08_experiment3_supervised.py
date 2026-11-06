"""Experiment III: character TF-IDF classifiers trained on the human reference labels.

Design
------
* Three classifiers (multinomial naive Bayes, linear SVM, MLP) on character 2-5-gram TF-IDF features,
  with fixed hyperparameters (no search), trained separately for each representation (original
  expression; English interpretation) on that representation's human reference labels.
* Five-fold cross-validation with folds formed by RESPONDENT (GroupKFold on Participant_ID): no respondent
  ever appears in both the training and the test part of a fold. The same folds are used for every
  model and both representations, so every model is scored on exactly the same 3,741 records, once each.
* Two training regimes on those same folds:
    1. Participant-disjoint: train on all records of the training respondents.
    2. Participant- and text-disjoint: as 1, but every training record whose normalised original text OR
       normalised English interpretation also occurs among the test respondents is removed from training.
       No test text is therefore ever seen in training, in either representation, while the test sets
       stay identical (all 3,741 records) across models and representations.
* Metrics: accuracy, macro-F1, balanced accuracy, Cohen's kappa; participant-cluster percentile bootstrap
  intervals (1,000 replicates, seed 42) for accuracy and macro-F1.

Run:  python run_experiment3_supervised.py ANNOTATION_CSV [OUT_DIR]
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

ANN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Human_Sentiment_Annotation_3741.csv")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("experiment3_supervised_results")
OUT.mkdir(parents=True, exist_ok=True)
LABELS = ["Negative", "Neutral", "Positive"]
SEED, REPS, K = 42, 1000, 5
MODELS = ["TFIDF_MultinomialNB", "TFIDF_LinearSVM", "TFIDF_MLP"]
TEXT = {"Original": "Original_Expression", "English": "English_Interpretation"}


def canonical(raw):
    m = {"negative": "Negative", "neg": "Negative", "neutral": "Neutral", "neu": "Neutral",
         "positive": "Positive", "pos": "Positive", "po": "Positive"}
    s = str(raw).strip().lower()
    if s not in m:
        raise ValueError(f"Unknown label {raw!r}")
    return m[s]


def model(name):
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True, max_features=20000,
                          min_df=2, max_df=0.98, dtype=np.float32)
    est = {"TFIDF_MultinomialNB": MultinomialNB(alpha=1.0),
           "TFIDF_LinearSVM": LinearSVC(C=1.0, class_weight="balanced", random_state=SEED),
           "TFIDF_MLP": MLPClassifier(hidden_layer_sizes=(64,), activation="relu", alpha=0.005, max_iter=120,
                                      early_stopping=True, n_iter_no_change=8, random_state=SEED)}[name]
    return make_pipeline(vec, est)


def norm(t):
    return re.sub(r"\s+", " ", str(t).strip().lower())


def cluster_ci(df, stat):
    pids = df.Participant_ID.unique()
    groups = {p: g for p, g in df.groupby("Participant_ID")}
    rng = np.random.default_rng(SEED)
    vals = [stat(pd.concat([groups[p] for p in rng.choice(pids, size=len(pids), replace=True)], ignore_index=True))
            for _ in range(REPS)]
    return np.quantile(vals, [0.025, 0.975])


def metrics(df):
    acc = lambda d: 100 * accuracy_score(d.Reference, d.Predicted)
    mf1 = lambda d: f1_score(d.Reference, d.Predicted, labels=LABELS, average="macro", zero_division=0)
    a_lo, a_hi = cluster_ci(df, acc)
    f_lo, f_hi = cluster_ci(df, mf1)
    return dict(N=len(df), Respondents=df.Participant_ID.nunique(), Accuracy=acc(df), Acc_CI_low=a_lo,
                Acc_CI_high=a_hi, Macro_F1=mf1(df), MF1_CI_low=f_lo, MF1_CI_high=f_hi,
                Balanced_Acc=100 * balanced_accuracy_score(df.Reference, df.Predicted),
                Kappa=cohen_kappa_score(df.Reference, df.Predicted, labels=LABELS))


a = pd.read_csv(ANN, dtype=str, keep_default_na=False, encoding="utf-8-sig")
for rep in TEXT:
    a[f"Human_{rep}_Sentiment"] = a[f"Human_{rep}_Sentiment"].map(canonical)
NO, NE = a.Original_Expression.map(norm), a.English_Interpretation.map(norm)

folds = list(GroupKFold(n_splits=K).split(a, groups=a.Participant_ID))
fold_of = np.zeros(len(a), dtype=int)
log = []
REGIMES = {"Participant-disjoint": [], "Participant- and text-disjoint": []}
for f, (tr, te) in enumerate(folds):
    fold_of[te] = f
    assert not set(a.Participant_ID.iloc[tr]) & set(a.Participant_ID.iloc[te])  # respondents never shared
    keep = tr[~(NO.iloc[tr].isin(set(NO.iloc[te])).to_numpy() | NE.iloc[tr].isin(set(NE.iloc[te])).to_numpy())]
    assert not set(NO.iloc[keep]) & set(NO.iloc[te]) and not set(NE.iloc[keep]) & set(NE.iloc[te])
    REGIMES["Participant-disjoint"].append(tr)
    REGIMES["Participant- and text-disjoint"].append(keep)
    log.append(dict(Fold=f, Test_records=len(te), Test_respondents=a.Participant_ID.iloc[te].nunique(),
                    Train_records_participant_disjoint=len(tr), Train_records_text_disjoint=len(keep),
                    Train_respondents_text_disjoint=a.Participant_ID.iloc[keep].nunique()))
pd.DataFrame(log).to_csv(OUT / "fold_summary.csv", index=False)
print(pd.DataFrame(log).to_string(index=False))

preds, rows = [], []
for regime, trains in REGIMES.items():
    for rep, col in TEXT.items():
        X, y = a[col], a[f"Human_{rep}_Sentiment"]
        for name in MODELS:
            yhat = np.empty(len(a), dtype=object)
            for (_, te), tr in zip(folds, trains):
                yhat[te] = model(name).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
            d = pd.DataFrame({"Expression_ID": a.Expression_ID, "Participant_ID": a.Participant_ID, "Regime": regime,
                              "Model": name, "Representation": rep, "Fold": fold_of, "Reference": y, "Predicted": yhat})
            preds.append(d)
            rows.append(dict(Regime=regime, Model=name, Representation=rep, **metrics(d)))
            print(regime, name, rep, f"acc={rows[-1]['Accuracy']:.1f} mf1={rows[-1]['Macro_F1']:.3f}")
for rep in TEXT:
    y = a[f"Human_{rep}_Sentiment"]
    rows.append(dict(Regime="Both", Model="Majority class", Representation=rep,
                     **metrics(pd.DataFrame({"Participant_ID": a.Participant_ID, "Reference": y,
                                             "Predicted": y.value_counts().idxmax()}))))
pd.concat(preds).to_csv(OUT / "experiment3_oof_predictions.csv", index=False)
pd.DataFrame(rows).round(3).to_csv(OUT / "experiment3_results.csv", index=False)
print("Saved to", OUT)
