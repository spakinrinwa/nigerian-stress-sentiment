"""Write a label-only file that contains no text, so that Experiments I and II can be checked without the restricted data.

Run only after the ethics committee has agreed that a label-only file may be published (see README).
    python tools/make_public_labels.py          -> data/public/labels_and_predictions.csv

One row per record. Columns: record and respondent identifiers (questionnaire row order, no personal data),
an integer pair identifier shared by identical (original, interpretation) pairs, the annotated language category,
referent, code-switching flag, analysis flags, the two harmonised reference labels and the label of every system
in both representations. No text, no text hashes (short expressions could be recovered from a hash), no safety
flags and no demographics are written.
"""
import os
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA_DIR", os.path.join(HERE, "..", "data", "restricted"))
OUT = os.environ.get("OUT_DIR", os.path.join(HERE, "..", "outputs"))
b = pd.read_csv(os.path.join(DATA, "sentiment_benchmark.csv"), dtype={"Expression_ID": str, "Participant_ID": str},
                keep_default_na=False, encoding="utf-8-sig")
h = pd.read_csv(os.path.join(OUT, "Human_Sentiment_Annotation_3741_harmonised.csv"), dtype=str, keep_default_na=False)
p = pd.read_csv(os.path.join(OUT, "predictions.csv"), dtype={"Expression_ID": str}, low_memory=False)
g = pd.read_csv(os.path.join(OUT, "gloss_flag.csv"), dtype=str)

def referent(prompt, code):
    p_ = prompt.strip().lower()
    if p_ in {"cultural_visible_stress_not_coping", "cultural_mental_exhaustion_burnout"} or int(code) == 400:
        return "Other observation"
    if p_ in {"cultural_exam_stress", "cultural_study_fatigue_deadline_pressure"}:
        return "Self scenario"
    return "Cultural lexicon"
lang = {"english": "English", "pidgin": "Nigerian Pidgin", "yoruba": "Yoruba", "yoruba-exclamation": "Yoruba exclamation",
        "unclear": "Unclear", "pidgin/english": "Pidgin/English", "nigerian english": "Nigerian English",
        "general nigerian usage": "General Nigerian usage"}
pair = (b.Original_Expression.str.strip() + "\x1f" + b.English_Interpretation.str.strip())
out = pd.DataFrame({
    "Expression_ID": b.Expression_ID, "Participant_ID": b.Participant_ID,
    "Pair_ID": pair.map({k: i for i, k in enumerate(pair.drop_duplicates())}),
    "Language_Category": b.Language.str.strip().str.lower().map(lang).fillna("Other or mixed"),
    "Referent": [referent(x, c) for x, c in zip(b.Prompt_Source, b.Priority)],
    "Code_Switched": b.Code_Switched, "Non_Interpretable_Code_999": (b.Priority.astype(int) == 999).astype(int),
    "Emoji_Corrupted_Original": b.Original_Expression.str.contains("Ã|Â", regex=True).astype(int)})
eng = h.set_index("Expression_ID").English_Interpretation.str.strip()
out["Stale_English_Input"] = [int(t.strip() != eng[e]) for e, t in zip(b.Expression_ID, b.English_Interpretation)]
out = out.merge(g.rename(columns={"Gloss": "Gloss_Type_Interpretation"}), on="Expression_ID")
out = out.merge(h[["Expression_ID", "Human_Original_Sentiment", "Human_English_Sentiment"]]
                .rename(columns={"Human_Original_Sentiment": "Reference_Original", "Human_English_Sentiment": "Reference_English"}),
                on="Expression_ID")
p["col"] = p.Model + "__" + p.Prompt_Mode + "__" + p.Representation
wide = p.pivot_table(index="Expression_ID", columns="col", values="Predicted_Sentiment", aggfunc="first").reset_index()
out = out.merge(wide, on="Expression_ID")
os.makedirs(os.path.join(HERE, "..", "data", "public"), exist_ok=True)
out.to_csv(os.path.join(HERE, "..", "data", "public", "labels_and_predictions.csv"), index=False)
print(out.shape)
