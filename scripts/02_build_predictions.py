"""Assemble one prediction file (outputs/predictions.csv) from the raw inference outputs and the harmonised reference labels.

Inputs (restricted data folder, see data/README.md):
  predictions_raw/<Model>__<Representation>__<Mode>.csv  from inference/run_classifiers_and_textblob.py
  qwen/Qwen_<Representation>_<ZeroShot|Cultural>.csv     from inference/qwen_label_scoring_colab.py
  sentiment_benchmark.csv                                 the texts the models scored (for the SHA-256 check)
and outputs/Human_Sentiment_Annotation_3741_harmonised.csv from script 01.
Every prediction is matched to its record through the SHA-256 hash of the input text; a mismatch stops the script.
"""
import glob, hashlib, os
import pandas as pd
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = lambda f: os.path.join(os.environ.get('DATA_DIR', os.path.join(_HERE, '..', 'data', 'restricted')), f)
OUTP = lambda f: os.path.join(os.environ.get('OUT_DIR', os.path.join(_HERE, '..', 'outputs')), f)

FAMILY = {"Cardiff_English": "Pretrained_Transformer", "Cardiff_Multilingual": "Pretrained_Transformer",
          "AfriSenti_AfroXLMR": "Pretrained_Transformer", "TextBlob_Lexicon": "Lexicon",
          "Qwen_Qwen2_5-3B-Instruct": "Instruction_LLM"}
COL = {"Original": "Original_Expression", "English": "English_Interpretation"}

b = pd.read_csv(DATA("sentiment_benchmark.csv"), dtype={"Expression_ID": str, "Participant_ID": str},
                keep_default_na=False, encoding="utf-8-sig")
sha = {rep: dict(zip(b.Expression_ID, b[c].astype(str).str.strip().map(lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest())))
       for rep, c in COL.items()}

parts = []
for f in sorted(glob.glob(DATA(os.path.join("predictions_raw", "*__*__*.csv")))):
    d = pd.read_csv(f, dtype={"Expression_ID": str, "Participant_ID": str}, keep_default_na=False)
    d["Prediction_File"] = os.path.basename(f)
    parts.append(d)
for f in sorted(glob.glob(DATA(os.path.join("qwen", "Qwen_*_*.csv")))):
    q = pd.read_csv(f, dtype={"Expression_ID": str, "Participant_ID": str}, keep_default_na=False)
    parts.append(pd.DataFrame({
        "Expression_ID": q.Expression_ID, "Participant_ID": q.Participant_ID, "Model": "Qwen_Qwen2_5-3B-Instruct",
        "Checkpoint": "Qwen/Qwen2.5-3B-Instruct", "Representation": q.Representation, "Prompt_Mode": q.Prompt_Mode,
        "Predicted_Sentiment": q.Generated_Label, "P_Negative": q.P_Negative, "P_Neutral": q.P_Neutral,
        "P_Positive": q.P_Positive, "Input_SHA256": q.Input_SHA256, "Raw_Output": q.Generated_Raw,
        "Scored_Label": q.Scored_Label, "Prediction_File": os.path.basename(f)}))
j = pd.concat(parts, ignore_index=True)
assert len(parts) == 12 and len(j) == 12 * len(b), (len(parts), len(j))

bad = [(e, r) for e, r, h in zip(j.Expression_ID, j.Representation, j.Input_SHA256) if sha[r][e] != h]
assert not bad, f"{len(bad)} predictions do not match the scored text, e.g. {bad[:3]}"
assert set(j.Predicted_Sentiment) <= {"Negative", "Neutral", "Positive"}, set(j.Predicted_Sentiment)

h = pd.read_csv(OUTP("Human_Sentiment_Annotation_3741_harmonised.csv"), dtype=str, keep_default_na=False).set_index("Expression_ID")
j["Human_Label"] = [h.at[e, "Human_Original_Sentiment"] if r == "Original" else h.at[e, "Human_English_Sentiment"]
                    for e, r in zip(j.Expression_ID, j.Representation)]
j["Correct"] = j.Human_Label == j.Predicted_Sentiment
j["Model_Family"] = j.Model.map(FAMILY)
for c in ["Language", "Construct", "Referent_Type", "Code_Switched"]:  # re-attached from the benchmark by later scripts
    j[c] = "Unspecified"
j.to_csv(OUTP("predictions.csv"), index=False)
print("predictions:", len(j), "rows;", j.groupby(["Model", "Prompt_Mode", "Representation"]).size().to_dict())
