"""Qwen2.5-3B-Instruct inference: generated labels and candidate scoring (Section 3.8 of the paper).

The prompts are those of Section 3.8. Only the input text is truncated (at 128 tokens); the instruction and the
final "Label:" line are never truncated. Two labels are saved for every record:
  a. Generation   : greedy decoding, at most 12 new tokens, strict parser (unparsed outputs are kept as "Unparsed").
  b. Candidate    : each label word is a single token; its probability at the answer position is renormalised over
     scoring        the three labels, and the highest gives the label. These probabilities define the near-ties.
The columns Original_Run_Prompt_Tokens and Original_Run_Truncated record the length of the full prompt; they are
diagnostics and are not used in the analysis.

HOW TO RUN ON GOOGLE COLAB
  1. Runtime > Change runtime type > T4 GPU.
  2. Upload sentiment_benchmark.csv and this script.
  3. In a cell:   !python qwen_label_scoring_colab.py
     Results are saved every 100 records; if Colab disconnects, run the same cell again and it resumes.
  4. Download the output folder and copy its four CSV files to data/restricted/qwen/.
"""
import hashlib, os, platform, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

INPUT = Path("sentiment_benchmark.csv")
OUT = Path("qwen_outputs")
CHECKPOINT = "Qwen/Qwen2.5-3B-Instruct"
MODES = ["ZeroShot", "Cultural"]
REPS = {"Original": "Original_Expression", "English": "English_Interpretation"}
TEXT_MAX_TOKENS = 128     # limit on the input text only, as for the classifiers
OLD_PROMPT_LIMIT = 128    # diagnostic only: flags prompts longer than 128 tokens
MAX_NEW_TOKENS = 12
LABELS = ["Negative", "Neutral", "Positive"]
SAVE_EVERY = 100
OUT.mkdir(exist_ok=True)


def prompt_for(text, mode, representation):  # the prompts of Section 3.8
    base = ("Classify only the sentiment of the text as Negative, Neutral, or Positive. "
            "Return exactly one label, no explanation. Sentiment describes the expression, "
            "not the respondent's mental-health status.\n")
    if mode == "Cultural":
        base += ("The text may contain Nigerian Pidgin, Yoruba, Hausa, Nigerian English, "
                 "idioms, slang or code-switching; interpret idiomatic and pragmatic meaning.\n")
    base += ("Input is a researcher-prepared English interpretation.\n" if representation == "English" else
             "Input is the original cultural expression.\n")
    return base + "Text: " + str(text) + "\nLabel:"


def normalize_label(s):  # strict label parser
    s = str(s).strip().strip("`\"' .:;,\n\t").lower()
    return {"negative": "Negative", "neg": "Negative", "neutral": "Neutral", "neu": "Neutral",
            "positive": "Positive", "pos": "Positive"}.get(s)


def main():
    assert torch.cuda.is_available(), "Select a GPU runtime (Runtime > Change runtime type > T4 GPU)."
    torch.manual_seed(42)
    df = pd.read_csv(INPUT, dtype={"Expression_ID": str, "Participant_ID": str}, keep_default_na=False)
    for c in REPS.values():
        df[c] = df[c].astype(str).str.strip()
    tok = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, torch_dtype=torch.float16, device_map="cuda").eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    label_ids = [tok(l, add_special_tokens=False).input_ids for l in LABELS]
    first = [ids[0] for ids in label_ids]
    first_token_ok = len(set(first)) == 3   # if the three labels start with different tokens
    (OUT / "run_info.txt").write_text(
        f"checkpoint={CHECKPOINT}\ntransformers={transformers.__version__}\ntorch={torch.__version__}\n"
        f"python={platform.python_version()}\ngpu={torch.cuda.get_device_name(0)}\n"
        f"label_token_ids={label_ids}\nfirst_token_distinct={first_token_ok}\n"
        f"started={time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
    print("label token ids:", label_ids, "| first tokens distinct:", first_token_ok)

    for rep, col in REPS.items():
        for mode in MODES:
            path = OUT / f"Qwen_{rep}_{mode}.csv"
            done = pd.read_csv(path, dtype={"Expression_ID": str}) if path.exists() else pd.DataFrame()
            rows = done.to_dict("records")
            seen = set(done.Expression_ID) if len(done) else set()
            todo = df[~df.Expression_ID.isin(seen)]
            print(f"{rep} {mode}: {len(seen)} done, {len(todo)} to go")
            t0 = time.time()
            for k, (_, r) in enumerate(todo.iterrows(), 1):
                text = r[col]
                text_ids = tok(text, add_special_tokens=False).input_ids
                text_cut = len(text_ids) > TEXT_MAX_TOKENS
                if text_cut:
                    text = tok.decode(text_ids[:TEXT_MAX_TOKENS])
                chat = tok.apply_chat_template([{"role": "user", "content": prompt_for(text, mode, rep)}],
                                               tokenize=False, add_generation_prompt=True)
                p_ids = tok(chat, add_special_tokens=False).input_ids
                # full-prompt length with untruncated text (diagnostic only)
                old_chat = tok.apply_chat_template([{"role": "user", "content": prompt_for(r[col], mode, rep)}],
                                                   tokenize=False, add_generation_prompt=True)
                old_len = len(tok(old_chat, add_special_tokens=False).input_ids)
                x = torch.tensor([p_ids], device="cuda")
                with torch.inference_mode():
                    # a. generation, no truncation of the prompt
                    g = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=MAX_NEW_TOKENS,
                                       do_sample=False, pad_token_id=tok.pad_token_id)
                    raw = tok.decode(g[0][x.shape[1]:], skip_special_tokens=True).strip()
                    gen_label = normalize_label(raw)
                    # b. label scoring: full-sequence log-probability of each label after the prompt
                    seqs = [p_ids + ids for ids in label_ids]
                    L = max(map(len, seqs))
                    batch = torch.full((3, L), tok.pad_token_id, device="cuda")
                    att = torch.zeros((3, L), dtype=torch.long, device="cuda")
                    for i, s in enumerate(seqs):
                        batch[i, :len(s)] = torch.tensor(s, device="cuda"); att[i, :len(s)] = 1
                    logp = torch.log_softmax(model(batch, attention_mask=att).logits.float(), -1)
                    seq_lp = []
                    for i, ids in enumerate(label_ids):
                        pos = range(len(p_ids) - 1, len(p_ids) - 1 + len(ids))
                        seq_lp.append(sum(logp[i, p, t].item() for p, t in zip(pos, ids)))
                    first_lp = [logp[0, len(p_ids) - 1, t].item() for t in first]
                score = first_lp if first_token_ok else seq_lp
                prob = np.exp(np.array(score) - np.max(score)); prob /= prob.sum()
                rows.append(dict(
                    Expression_ID=r.Expression_ID, Participant_ID=r.Participant_ID, Representation=rep,
                    Prompt_Mode=mode, Input_SHA256=hashlib.sha256(r[col].encode("utf-8")).hexdigest(),
                    Prompt_Tokens=len(p_ids), Original_Run_Prompt_Tokens=old_len,
                    Original_Run_Truncated=int(old_len > OLD_PROMPT_LIMIT), Text_Truncated=int(text_cut),
                    Generated_Raw=raw, Generated_Label=gen_label or "Unparsed",
                    Scored_Label=LABELS[int(np.argmax(score))],
                    P_Negative=prob[0], P_Neutral=prob[1], P_Positive=prob[2],
                    LogP_Seq_Negative=seq_lp[0], LogP_Seq_Neutral=seq_lp[1], LogP_Seq_Positive=seq_lp[2],
                    Scoring=("first_token" if first_token_ok else "full_sequence")))
                if k % SAVE_EVERY == 0 or k == len(todo):
                    pd.DataFrame(rows).to_csv(path, index=False)
                    rate = k / (time.time() - t0)
                    print(f"  {rep} {mode}: {len(rows)}/{len(df)}  ({rate:.1f} records/s)", flush=True)
    print("Finished. Download the folder:", OUT.resolve())


if __name__ == "__main__":
    main()
