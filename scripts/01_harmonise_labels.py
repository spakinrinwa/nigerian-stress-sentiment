"""Make reference labels consistent across identical (original, interpretation) pairs.
Rule: every record of a repeated pair takes the label most of its records have; ties were settled by the author (TIES)."""
import pandas as pd, sys
SRC = sys.argv[1] if len(sys.argv) > 1 else "Human_Sentiment_Annotation_3741.csv"
a = pd.read_csv(SRC, dtype=str, keep_default_na=False, encoding="utf-8-sig")
canon = {"negative": "Negative", "neutral": "Neutral", "positive": "Positive", "po": "Positive"}
COLS = {"Original": "Human_Original_Sentiment", "English": "Human_English_Sentiment"}
for c in COLS.values(): a[c] = a[c].str.strip().str.lower().map(canon)
assert a[list(COLS.values())].notna().all().all()
a["key"] = a.Original_Expression.str.strip() + "||" + a.English_Interpretation.str.strip()
# author decisions for tied pairs (original text, interpretation) -> {side: label}
TIES = [
 ("don tire", "I am tired or exhausted.", {"Original": "Negative", "English": "Negative"}),
 ("he don tire", "I am tired or exhausted.", {"Original": "Negative", "English": "Negative"}),
 ("i dan tire", "I am tired or exhausted.", {"Original": "Negative"}),
 ("overthinking", "I am overthinking.", {"Original": "Negative", "English": "Negative"}),
 ("i dey overthink", "I am overthinking.", {"Original": "Negative"}),
 ("i am overthinking / mentally stressed", "I am overthinking.", {"Original": "Negative"}),
 ("e don do", "That is enough / I am fed up.", {"Original": "Negative"}),
 ("hmm", "A sigh/thoughtful sound indicating concern, stress, or uncertainty.", {"Original": "Neutral"}),
 ("guyy", "No interpretable cultural expression", {"Original": "Neutral"}),
 ("i can't wait to finish this exam", "I am experiencing academic pressure or difficulty.", {"Original": "Negative", "English": "Negative"}),
 ("i can't kill myself", "I cannot kill myself", {"Original": "Negative", "English": "Negative"}),
 ("i'm nervous", "I feel nervous.", {"English": "Neutral"}),
 ("im tied", "I am tired", {"Original": "Negative", "English": "Negative"}),
 ("ore mi", "I am tired", {"Original": "Negative", "English": "Negative"}),
]
log = []
def setlab(idx, side, new, reason):
    col = COLS[side]
    for i in idx:
        if a.at[i, col] != new:
            log.append(dict(Expression_ID=a.at[i, "Expression_ID"], Side=side, Old=a.at[i, col], New=new, Reason=reason)); a.at[i, col] = new
decided = set()
for o, e, d in TIES:
    idx = a.index[(a.Original_Expression.str.strip() == o) & (a.English_Interpretation.str.strip() == e)]
    keys = set(a.loc[idx, "key"]); assert len(keys) == 1 and len(idx) >= 2, (o, len(idx), keys)
    for side, lab in d.items():
        setlab(idx, side, lab, "tie or author decision"); decided.add((a.at[idx[0], "key"], side))
for k, g in a.groupby("key"):
    if len(g) < 2: continue
    for side, col in COLS.items():
        if (k, side) in decided or g[col].nunique() == 1: continue
        vc = g[col].value_counts()
        assert vc.iloc[0] > vc.iloc[1], ("unresolved tie", k, side, vc.to_dict())
        setlab(g.index, side, vc.index[0], "majority")
L = pd.DataFrame(log); L.to_csv("label_harmonisation_log.csv", index=False)
chk = a.groupby("key")[list(COLS.values())].nunique()
assert (chk <= 1).all().all()
a.drop(columns="key").to_csv("Human_Sentiment_Annotation_3741_harmonised.csv", index=False)
print(L.groupby(["Side", "Reason"]).size(), "\nrecords changed:", L.Expression_ID.nunique(), "labels changed:", len(L))
