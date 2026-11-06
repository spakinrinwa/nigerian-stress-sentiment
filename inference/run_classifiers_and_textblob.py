"""
Inference for the three pretrained classifiers and the TextBlob lexicon (Experiment I inputs).
Put sentiment_benchmark.csv beside this script and run it (python run_classifiers_and_textblob.py).
Predictions are written to outputs/predictions/<Model>__<Representation>__<Mode>.csv; copy them to
data/restricted/predictions_raw/ before running the analysis scripts. No reference labels are read.
"""
from __future__ import annotations
import csv
import hashlib
import itertools
import json
import logging
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
INPUT = HERE / "sentiment_benchmark.csv"
ROOT = HERE / "outputs"
PRED = ROOT / "predictions"
REPORTS = ROOT / "reports"
for folder in (PRED, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

# -------- EDIT THESE SETTINGS; F5 RUNS EVERYTHING AUTOMATICALLY --------
MODELS = {
    "Cardiff_English": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "Cardiff_Multilingual": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "AfriSenti_AfroXLMR": "Davlan/afrisenti-twitter-sentiment-afroxlmr-large",
}
# Optional second family. Requires capable GPU and transformers/accelerate.
# Set to True only after checking memory, access rights and license.
RUN_LLM = False  # The Qwen results in the paper come from qwen_label_scoring_colab.py, not from this script.
LLM_CHECKPOINTS = ["Qwen/Qwen2.5-3B-Instruct"]
LLM_MODES = ["ZeroShot", "Cultural"]
RUN_TRANSFORMERS = True
RUN_PSEUDOLABEL_STUDENTS = False  # not used in the paper
RUN_LEXICON_CONTROL = True
BATCH_SIZE = 12
MAX_TOKENS = 128
LLM_MAX_NEW_TOKENS = 12
BOOTSTRAP_REPS = 1000
SEED = 42
MIN_SUBGROUP_N = 25
ALLOW_MODEL_DOWNLOAD = True
# ----------------------------------------------------------------------

LABELS = ["Negative", "Neutral", "Positive"]
KEYS = ["Expression_ID", "Participant_ID"]
logger = logging.getLogger("sentiment")

def utc():
    return datetime.now(timezone.utc).isoformat()

def safe_slug(s):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(s)).strip("_")[:90]

def normalize_label(label):
    s = str(label).strip().lower()
    if s in ("negative", "neg"):
        return "Negative"
    if s in ("neutral", "neu"):
        return "Neutral"
    if s in ("positive", "pos"):
        return "Positive"
    return None

def text_hash(s):
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()

def norm(s):
    return str(s).strip().lower().replace(" ", "_")

def referent_from_prompt(prompt, priority):
    # Locked canonical prompt mapping, not keyword guessing on the natural-language answer.
    p = str(prompt).strip().lower()
    self_prompts = {"cultural_exam_stress", "cultural_study_fatigue_deadline_pressure"}
    other_prompts = {"cultural_visible_stress_not_coping", "cultural_mental_exhaustion_burnout"}
    if p in other_prompts:
        return "Other_Observation"
    try:
        if int(float(priority)) == 400:
            return "Other_Observation"
    except (TypeError, ValueError):
        pass
    if p in self_prompts:
        return "Self_Scenario"
    if p.startswith("cultural_"):
        return "Cultural_Lexicon"
    raise ValueError(f"Unrecognized questionnaire prompt: {prompt!r}. Review prompt dictionary.")

def read_data():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing {INPUT}. Put the paired corpus beside this script.")
    df = pd.read_csv(INPUT, dtype={"Expression_ID": str, "Participant_ID": str}, keep_default_na=False)
    needed = ["Expression_ID", "Participant_ID", "Original_Expression", "English_Interpretation"]
    missing = sorted(set(needed)-set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    for name in ("Original_Expression", "English_Interpretation"):
        df[name] = df[name].astype(str).str.strip()
    if df["Expression_ID"].duplicated().any():
        raise ValueError("Expression_ID is not unique. Do not run a potentially mispaired benchmark.")
    df["Language"] = df.get("Language", pd.Series("", index=df.index)).replace("", "Unspecified")
    df["Construct"] = df.get("Construct", pd.Series("", index=df.index)).replace("", "Unspecified")
    df["Priority"] = df.get("Priority", pd.Series("", index=df.index))
    df["Prompt_Source"] = df.get("Prompt_Source", pd.Series("", index=df.index))
    # Always derive referent from the source prompt, never trust a stale benchmark column.
    df["Referent_Type"] = [referent_from_prompt(p, q) for p, q in
                           zip(df["Prompt_Source"], df["Priority"])]
    old_switch = df.get("Code_Switched", pd.Series("", index=df.index)).astype(str)
    df["Code_Switched"] = pd.to_numeric(old_switch, errors="coerce").fillna(0).astype(int)
    df["Pair_Eligible"] = df["Original_Expression"].ne("") & df["English_Interpretation"].ne("")
    df["Is_Noise_Priority"] = pd.to_numeric(df["Priority"], errors="coerce").eq(999)
    df["Original_SHA256"] = df["Original_Expression"].map(text_hash)
    df["English_SHA256"] = df["English_Interpretation"].map(text_hash)
    df.to_csv(REPORTS/"input_audit.csv", index=False)
    if not df["Pair_Eligible"].all():
        logger.warning("%d records are missing one text representation; excluding from paired inference.",
                       int((~df["Pair_Eligible"]).sum()))
    return df[df["Pair_Eligible"]].copy()

def model_label_map(model):
    idmap = model.config.id2label
    mapping = {int(k): normalize_label(v) for k, v in idmap.items()}
    if len(mapping) != 3 or set(mapping.values()) != set(LABELS):
        # Some checkpoints use LABEL_0/1/2. Do not guess a potentially false semantic mapping.
        raise ValueError(f"Cannot verify negative/neutral/positive mapping: {idmap}. "
                         "Inspect model card and add a justified explicit mapping.")
    return mapping

def done_path(model_name, representation, mode):
    return PRED / f"{safe_slug(model_name)}__{representation}__{mode}.csv"

def completed_rows(path, corpus, representation):
    if not path.exists():
        return {}
    old = pd.read_csv(path, dtype={"Expression_ID": str, "Participant_ID": str}, keep_default_na=False)
    if "Input_SHA256" not in old.columns:
        raise ValueError(f"Previous predictions lack input hash: {path}; archive/remove manually.")
    hashes = dict(zip(corpus["Expression_ID"], corpus["Original_SHA256" if representation=="Original" else "English_SHA256"]))
    valid={}
    for row in old.to_dict("records"):
        id_=row["Expression_ID"]
        if id_ in hashes and row["Input_SHA256"] == hashes[id_] and row.get("Predicted_Sentiment") in LABELS:
            valid[id_]=row
    return valid

def atomic_save(rows, path):
    temp=path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temp, index=False)
    temp.replace(path)

def classifier_run(df, shortname, checkpoint, representation):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as ex:
        logger.error("Install dependencies to run %s: %s", shortname, ex)
        return False
    key = "Original_Expression" if representation=="Original" else "English_Interpretation"
    hashkey = "Original_SHA256" if representation=="Original" else "English_SHA256"
    output=done_path(shortname,representation,"Classifier")
    completed=completed_rows(output,df,representation)
    pending=df[~df["Expression_ID"].isin(completed)]
    if pending.empty:
        logger.info("Already complete: %s / %s", shortname,representation)
        return True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading %s on %s (%d pending)",checkpoint,device,len(pending))
    kwargs={"local_files_only": not ALLOW_MODEL_DOWNLOAD}
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,**kwargs)
    model=AutoModelForSequenceClassification.from_pretrained(checkpoint,**kwargs).to(device)
    model.eval()
    mapping=model_label_map(model)
    records=list(completed.values())
    try:
        for start in range(0,len(pending),BATCH_SIZE):
            part=pending.iloc[start:start+BATCH_SIZE]
            enc=tokenizer(part[key].tolist(),truncation=True,padding=True,max_length=MAX_TOKENS,
                          return_tensors="pt",return_length=False)
            # separately flag potential truncation (full encoded length excludes padding)
            lengths=[len(tokenizer.encode(s,truncation=False)) for s in part[key]]
            enc={k:v.to(device) for k,v in enc.items()}
            with torch.inference_mode():
                probs=torch.softmax(model(**enc).logits,dim=-1).cpu().numpy()
            for (_,r),pr,ln in zip(part.iterrows(),probs,lengths):
                by={mapping[i]:float(pr[i]) for i in range(3)}
                values=np.array([by[l] for l in LABELS])
                records.append({
                    "Expression_ID":r["Expression_ID"], "Participant_ID":r["Participant_ID"],
                    "Model":shortname,"Checkpoint":checkpoint,"Representation":representation,
                    "Prompt_Mode":"Classifier","Predicted_Sentiment":LABELS[int(np.argmax(values))],
                    "P_Negative":by["Negative"],"P_Neutral":by["Neutral"],
                    "P_Positive":by["Positive"],"Confidence":float(max(values)),
                    "Entropy":float(-(np.clip(values,1e-12,1)*np.log(np.clip(values,1e-12,1))).sum()),
                    "Input_Tokens":ln,"Was_Truncated":int(ln>MAX_TOKENS),
                    "Input_SHA256":r[hashkey],"Run_UTC":utc(),"Error":""
                })
            atomic_save(records,output)
            logger.info("%s %s: %d/%d scored",shortname,representation,len(records),len(df))
        return True
    finally:
        del model
        if device=="cuda":
            torch.cuda.empty_cache()

def lexicon_run(df, representation):
    from textblob import TextBlob
    key="Original_Expression" if representation=="Original" else "English_Interpretation"
    hashkey="Original_SHA256" if representation=="Original" else "English_SHA256"
    output=done_path("TextBlob_Lexicon",representation,"Lexicon")
    rows=list(completed_rows(output,df,representation).values())
    known={r["Expression_ID"] for r in rows}
    for _,r in df.iterrows():
        if r["Expression_ID"] in known:
            continue
        score=float(TextBlob(r[key]).sentiment.polarity)
        pred="Positive" if score>0 else ("Negative" if score<0 else "Neutral")
        rows.append({
            "Expression_ID":r["Expression_ID"],"Participant_ID":r["Participant_ID"],
            "Model":"TextBlob_Lexicon","Checkpoint":"TextBlob_Pattern_Lexicon",
            "Representation":representation,"Prompt_Mode":"Lexicon",
            "Predicted_Sentiment":pred,"P_Negative":np.nan,"P_Neutral":np.nan,
            "P_Positive":np.nan,"Confidence":np.nan,"Entropy":np.nan,
            "Input_Tokens":np.nan,"Was_Truncated":0,"Input_SHA256":r[hashkey],
            "Run_UTC":utc(),"Error":""
        })
    atomic_save(rows,output)

def prompt_for(text, mode, representation):
    base=("Classify only the sentiment of the text as Negative, Neutral, or Positive. "
          "Return exactly one label, no explanation. Sentiment describes the expression, "
          "not the respondent's mental-health status.\n")
    if mode=="Cultural":
        base+=("The text may contain Nigerian Pidgin, Yoruba, Hausa, Nigerian English, "
               "idioms, slang or code-switching; interpret idiomatic and pragmatic meaning.\n")
    base+=("Input is a researcher-prepared English interpretation.\n" if representation=="English" else
           "Input is the original cultural expression.\n")
    return base + "Text: "+str(text)+"\nLabel:"

def parse_generated_label(text):
    # Only a single label (allow wrapping punctuation); no scanning for label mentioned in explanation.
    clean=str(text).strip().strip("`\"' .:;,\n\t").lower()
    return normalize_label(clean)

def llm_run(df, shortname, checkpoint, representation, mode):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    key="Original_Expression" if representation=="Original" else "English_Interpretation"
    hashkey="Original_SHA256" if representation=="Original" else "English_SHA256"
    path=done_path(shortname,representation,mode)
    completed=completed_rows(path,df,representation)
    pending=df[~df["Expression_ID"].isin(completed)]
    if pending.empty:
        return
    if not torch.cuda.is_available():
        logger.warning("LLM %s on CPU may take a very long time",checkpoint)
    tok=AutoTokenizer.from_pretrained(checkpoint,local_files_only=not ALLOW_MODEL_DOWNLOAD)
    model=AutoModelForCausalLM.from_pretrained(checkpoint,torch_dtype="auto",device_map="auto",
                                               local_files_only=not ALLOW_MODEL_DOWNLOAD).eval()
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    records=list(completed.values())
    for j,(_,r) in enumerate(pending.iterrows(),1):
        prompt=prompt_for(r[key],mode,representation)
        if tok.chat_template:
            prompt=tok.apply_chat_template([{"role":"user","content":prompt}],
                                           tokenize=False,add_generation_prompt=True)
        inputs=tok(prompt,return_tensors="pt",truncation=True,max_length=MAX_TOKENS).to(model.device)
        with torch.inference_mode():
            generated=model.generate(**inputs,max_new_tokens=LLM_MAX_NEW_TOKENS,do_sample=False,
                                     pad_token_id=tok.pad_token_id)
        raw=tok.decode(generated[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True).strip()
        label=parse_generated_label(raw)
        records.append({
            "Expression_ID":r["Expression_ID"],"Participant_ID":r["Participant_ID"],
            "Model":shortname,"Checkpoint":checkpoint,"Representation":representation,
            "Prompt_Mode":mode,"Predicted_Sentiment":label or "Unparsed",
            "Raw_Output":raw,"P_Negative":np.nan,"P_Neutral":np.nan,
            "P_Positive":np.nan,"Confidence":np.nan,"Entropy":np.nan,
            "Input_SHA256":r[hashkey],"Run_UTC":utc(),
            "Error":"" if label else "unparsed_label"
        })
        if j%10==0 or j==len(pending):
            atomic_save(records,path)
            logger.info("LLM %s %s %s %d/%d",shortname,representation,mode,j,len(pending))

def common_keys(a,b):
    aa=a[KEYS+["Predicted_Sentiment"]].rename(columns={"Predicted_Sentiment":"A"})
    bb=b[KEYS+["Predicted_Sentiment"]].rename(columns={"Predicted_Sentiment":"B"})
    return aa.merge(bb,on=KEYS,validate="one_to_one")

def cluster_ci(frame, predicate, reps=BOOTSTRAP_REPS):
    # Cluster bootstrap without costly repeated concatenation.
    rng=np.random.default_rng(SEED)
    group=frame.groupby("Participant_ID",sort=False)
    n=group.size().to_numpy(dtype=int)
    hits=group.apply(lambda g:int(predicate(g).sum()),include_groups=False).to_numpy(dtype=int)
    k=len(n)
    if not k:
        return np.nan,np.nan
    vals=np.empty(reps,dtype=float)
    for i in range(reps):
        ids=rng.integers(0,k,size=k)
        vals[i]=hits[ids].sum()/n[ids].sum()
    return np.quantile(vals,[.025,.975])

def analyze(df):
    allpred=[]
    for f in sorted(PRED.glob("*.csv")):
        a=pd.read_csv(f,dtype={"Expression_ID":str,"Participant_ID":str},keep_default_na=False)
        if a.empty:
            continue
        a=a.merge(df[KEYS],on=KEYS,how="inner",validate="one_to_one")
        allpred.append(a)
    if not allpred:
        logger.warning("No predictions yet; analysis skipped")
        return
    p=pd.concat(allpred,ignore_index=True)
    p=p[p["Predicted_Sentiment"].isin(LABELS)].copy()
    p.to_csv(REPORTS/"all_valid_predictions.csv",index=False)
    summary=[]
    for (model,rep,mode),g in p.groupby(["Model","Representation","Prompt_Mode"],sort=True):
        counts=g["Predicted_Sentiment"].value_counts()
        summary.append({"Model":model,"Representation":rep,"Mode":mode,"N":len(g),
                        "Participants":g["Participant_ID"].nunique(),
                        **{f"N_{x}":int(counts.get(x,0)) for x in LABELS},
                        **{f"Pct_{x}":100*float(counts.get(x,0))/len(g) for x in LABELS},
                        "Mean_Confidence":pd.to_numeric(g.get("Confidence"),errors="coerce").mean(),
                        "Mean_Entropy":pd.to_numeric(g.get("Entropy"),errors="coerce").mean()})
    pd.DataFrame(summary).to_csv(REPORTS/"sentiment_distributions.csv",index=False)
    conditions={(m,r,pm):g for (m,r,pm),g in p.groupby(["Model","Representation","Prompt_Mode"])}
    model_modes=sorted({(m,pm) for (m,r,pm) in conditions})
    paired=[]
    transitions=[]
    subgroup=[]
    examples=[]
    unique_sensitivity=[]
    meta=df[KEYS+["Language","Construct","Referent_Type","Code_Switched",
                  "Original_Expression","English_Interpretation"]]
    for m,pm in model_modes:
        if (m,"Original",pm) not in conditions or (m,"English",pm) not in conditions:
            continue
        d=common_keys(conditions[(m,"Original",pm)],conditions[(m,"English",pm)])
        d=d.merge(meta,on=KEYS,validate="one_to_one")
        if len(d)==0:
            continue
        d["Shift"]=d["A"].ne(d["B"])
        low,high=cluster_ci(d,lambda x:x["Shift"])
        paired.append({"Model":m,"Mode":pm,"N_Pairs":len(d),
                       "N_Participants":d["Participant_ID"].nunique(),
                       "Agreement":float(1-d["Shift"].mean()),"Shift_Rate":float(d["Shift"].mean()),
                       "Shift_CI_Low":low,"Shift_CI_High":high,
                       "Kappa":cohen_kappa_score(d["A"],d["B"],labels=LABELS)})
        for (a,b),g in d.groupby(["A","B"]):
            transitions.append({"Model":m,"Mode":pm,"Original_Label":a,
                                "English_Label":b,"Count":len(g)})
        for field in ["Language","Construct","Referent_Type","Code_Switched"]:
            for val,g in d.groupby(field,dropna=False):
                subgroup.append({"Model":m,"Mode":pm,"Subgroup_Type":field,"Subgroup":val,
                                 "N":len(g),"Participants":g["Participant_ID"].nunique(),
                                 "Shift_Rate":float(g["Shift"].mean()),
                                 "Inferential_Eligible":len(g)>=MIN_SUBGROUP_N})
        examples.append(d.loc[d["Shift"],KEYS+["A","B","Language","Construct","Referent_Type",
                                               "Original_Expression","English_Interpretation"]].assign(Model=m,Mode=pm))
        undup=d.drop_duplicates(["Original_Expression","English_Interpretation"])
        unique_sensitivity.append({"Model":m,"Mode":pm,"Unique_Pairs":len(undup),
                                   "Shift_Rate_Unique":undup["Shift"].mean()})
    cross=[]
    for rep in ("Original","English"):
        avail=[(m,pm) for (m,r,pm) in conditions if r==rep]
        for a,b in itertools.combinations(sorted(avail),2):
            first=conditions[(a[0],rep,a[1])]
            second=conditions[(b[0],rep,b[1])]
            d=common_keys(first,second)
            if len(d)==0:continue
            d["Disagree"]=d["A"].ne(d["B"])
            ci=cluster_ci(d,lambda x:x["Disagree"])
            cross.append({"Representation":rep,"Model_A":a[0],"Mode_A":a[1],
                          "Model_B":b[0],"Mode_B":b[1],"N":len(d),
                          "Agreement":1-d["Disagree"].mean(),
                          "Disagreement_CI_Low":ci[0],"Disagreement_CI_High":ci[1],
                          "Kappa":cohen_kappa_score(d["A"],d["B"],labels=LABELS)})
    pd.DataFrame(paired).to_csv(REPORTS/"paired_translation_agreement.csv",index=False)
    pd.DataFrame(transitions).to_csv(REPORTS/"original_english_transition_counts.csv",index=False)
    pd.DataFrame(subgroup).to_csv(REPORTS/"translation_shifts_by_subgroup.csv",index=False)
    pd.DataFrame(cross).to_csv(REPORTS/"cross_model_agreement.csv",index=False)
    pd.DataFrame(unique_sensitivity).to_csv(REPORTS/"unique_pair_sensitivity.csv",index=False)
    if examples:
        pd.concat(examples,ignore_index=True).to_csv(REPORTS/"translation_disagreement_examples.csv",index=False)
    logger.info("Analysis complete. Reports: %s",REPORTS)

def run():
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(ROOT/"run.log",encoding="utf-8")])
    d=read_data()
    logger.info("Paired corpus: %d expressions; %d participants",len(d),d["Participant_ID"].nunique())
    for rep in ("Original","English"):
        if RUN_LEXICON_CONTROL:
            lexicon_run(d,rep)
        if RUN_TRANSFORMERS:
            for name,checkpoint in MODELS.items():
                try:
                    classifier_run(d,name,checkpoint,rep)
                except Exception:
                    logger.exception("Failed %s %s. Other conditions continue.",name,rep)
    if RUN_LLM:
        for name in LLM_CHECKPOINTS:
            for rep in ("Original","English"):
                for pm in LLM_MODES:
                    try:
                        llm_run(d,safe_slug(name),name,rep,pm)
                    except Exception:
                        logger.exception("LLM failed %s %s %s",name,rep,pm)
    if RUN_PSEUDOLABEL_STUDENTS:
        try:
            from run_student_models import run_students
            run_students(d, PRED, REPORTS)
        except Exception:
            logger.exception("Pseudo-label student extension failed; other results remain available")
    analyze(d)
    manifest={"timestamp":utc(),"python":sys.version,"system":platform.platform(),
              "models":MODELS,"llm_enabled":RUN_LLM,"llm_ids":LLM_CHECKPOINTS,
              "bootstrap_reps":BOOTSTRAP_REPS,"seed":SEED,"max_tokens":MAX_TOKENS,
              "n_input":len(d),"n_participants":int(d["Participant_ID"].nunique()),
              "model_results_are_accuracy":False}
    try:
        import torch
        manifest["torch_version"]=torch.__version__
        manifest["cuda_available"]=torch.cuda.is_available()
    except ImportError:pass
    try:
        import transformers
        manifest["transformers_version"]=transformers.__version__
    except ImportError:pass
    (REPORTS/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
    logger.info("Finished. Outputs located at %s",ROOT)

if __name__=="__main__":
    run()
