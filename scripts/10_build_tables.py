"""Build every results table of the manuscript (Tables 4, 6, A.1-A.9 and the values behind Figs. 5 and 7) from the outputs of scripts 03-05."""
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = lambda f: os.path.join(os.environ.get('DATA_DIR', os.path.join(_HERE, '..', 'data', 'restricted')), f)
OUTP = lambda f: os.path.join(os.environ.get('OUT_DIR', os.path.join(_HERE, '..', 'outputs')), f)
import json, re, subprocess, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score
V = OUTP('')
L = ['Negative', 'Neutral', 'Positive']
SYS = {'RoBERTa': ('Cardiff_English', 'Classifier'), 'XLM-R': ('Cardiff_Multilingual', 'Classifier'),
       'AfriSenti': ('AfriSenti_AfroXLMR', 'Classifier'), 'Qwen zero-shot': ('Qwen_Qwen2_5-3B-Instruct', 'ZeroShot'),
       'Qwen cultural': ('Qwen_Qwen2_5-3B-Instruct', 'Cultural'), 'TextBlob': ('TextBlob_Lexicon', 'Lexicon')}
cond = lambda s, r: f"{SYS[s][0]} | {r} | {SYS[s][1]}"
f1 = lambda x: f"{x:.1f}"; f3 = lambda x: f"{x:.3f}"; fn = lambda x: f"{int(x):,}"
sg = lambda x: ("+" if x >= 0 else "−") + f"{abs(x):.1f}"
ci = lambda lo, hi: f"[{lo:.1f}, {hi:.1f}]".replace('-', '−')
t8 = pd.read_csv(V + 'T8_performance_full.csv').set_index('Condition')
mf = pd.read_csv(V + 'macroF1_CI_1000.csv').set_index('Cond')
dm = pd.read_csv(V + 'T8b_delta_m.csv'); dm['S'] = [next(k for k, v in SYS.items() if v == (m, md)) for m, md in zip(dm.Model, dm.Mode)]; dm = dm.set_index('S')
sh = pd.read_csv(V + 'T6_shifts.csv')
t10 = pd.read_csv(V + 'T10_subgroups.csv'); t11 = pd.read_csv(V + 'T11_sensitivity.csv')
h = pd.read_csv(V + 'Human_Sentiment_Annotation_3741_harmonised.csv', dtype=str, keep_default_na=False)
O, E = h.Human_Original_Sentiment, h.Human_English_Sentiment
T = {}
# ---- Table 4
ct = pd.crosstab(O, E).reindex(index=L, columns=L, fill_value=0)
rows = [['Original \\ English'] + L + ['Total']]
for r in L: rows.append([r] + [fn(ct.loc[r, c]) for c in L] + [fn(ct.loc[r].sum())])
rows.append(['Total'] + [fn(ct[c].sum()) for c in L] + [fn(ct.values.sum())]); T['Table 4'] = rows
# ---- baseline
base = {r: (100 * (lab == 'Negative').mean(), f1_score(lab, ['Negative'] * len(lab), labels=L, average='macro', zero_division=0)) for r, lab in [('Original', O), ('English', E)]}
# ---- Table 6
rows = [['System', 'Accuracy % O / E', 'Gain (percentage points)', 'Macro-F1 O / E', 'Balanced accuracy % O / E', 'κ O / E']]
for s in SYS:
    o, e = t8.loc[cond(s, 'Original')], t8.loc[cond(s, 'English')]
    rows.append([s, f"{f1(o.Accuracy)} / {f1(e.Accuracy)}", sg(e.Accuracy - o.Accuracy), f"{f3(o.Macro_F1)} / {f3(e.Macro_F1)}",
                 f"{f1(o.Balanced_Acc)} / {f1(e.Balanced_Acc)}", f"{f3(o.Kappa)} / {f3(e.Kappa)}"])
rows.append(['Baseline (always Negative)', f"{f1(base['Original'][0])} / {f1(base['English'][0])}", sg(base['English'][0] - base['Original'][0]),
             f"{f3(base['Original'][1])} / {f3(base['English'][1])}", '33.3 / 33.3', '0.000 / 0.000'])
T['Table 6'] = rows
# ---- Table A.1
d5 = pd.read_csv(V + 'T5_distributions.csv').set_index('Cond')
rows = [['System', 'Representation', 'N valid', 'Negative', 'Neutral', 'Positive']]
for s in SYS:
    for r in ['Original', 'English']:
        x = d5.loc[cond(s, r)]; rows.append([s, r, fn(x.N), f1(x.Negative), f1(x.Neutral), f1(x.Positive)])
for r, lab in [('Original', O), ('English', E)]:
    v = lab.value_counts(normalize=True) * 100; rows.append(['Reference labels', r, '3,741'] + [f1(v.get(c, 0)) for c in L])
T['Table A.1'] = rows
# ---- Table A.2 (reference row)
agree = (O == E)
pid = h.Participant_ID
g = pd.DataFrame({'a': agree.astype(float), 'p': pid}).groupby('p').a.agg(['sum', 'count'])
rng = np.random.default_rng(42); ids = rng.integers(0, len(g), size=(1000, len(g)))
bs = g['sum'].values[ids].sum(1) / g['count'].values[ids].sum(1) * 100
alo, ahi = np.quantile(bs, [.025, .975])
T['ref'] = dict(agree=100 * agree.mean(), n_agree=int(agree.sum()), lo=alo, hi=ahi, kappa=cohen_kappa_score(O, E, labels=L),
                O=O.value_counts().to_dict(), E=E.value_counts().to_dict())
# ---- Table A.3
rows = [['System', 'Repr.', 'Negative P / R', 'Neutral P / R', 'Positive P / R']]
for s in SYS:
    for r in ['Original', 'English']:
        x = t8.loc[cond(s, r)]
        rows.append([s, r] + [f"{f3(x['P_' + c])} / {f3(x['R_' + c])}" for c in L])
T['Table A.3'] = rows
# ---- Table A.9
rows = [['System', 'Repr.', 'N (coverage %)', 'Respondents', 'Accuracy % [95% CI]', 'Macro-F1 [95% CI]', 'Balanced acc. %', 'κ']]
for s in SYS:
    for r in ['Original', 'English']:
        c = cond(s, r); x = t8.loc[c]
        rows.append([s, r, '3,741 (100.0)', '314', f"{f1(x.Accuracy)} {ci(x.Acc_CI_low, x.Acc_CI_high)}",
                     f"{f3(x.Macro_F1)} [{mf.loc[c].lo:.3f}, {mf.loc[c].hi:.3f}]", f1(x.Balanced_Acc), f3(x.Kappa)])
for r in ['Original', 'English']:
    rows.append(['Majority class (always Negative)', r, '3,741 (100.0)', '314', f1(base[r][0]), f3(base[r][1]), '33.3', '0.000'])
T['Table A.9'] = rows
json.dump(T, open(V + 'tables.json', 'w'), indent=0, default=float)
print(json.dumps({k: v for k, v in T.items() if k in ('Table 4', 'Table 6', 'ref')}, indent=0, default=float))
# ---- Table A.2 reference row values
T['ref_shift'] = dict(shift=100 - T['ref']['agree'], lo=100 - T['ref']['hi'], hi=100 - T['ref']['lo'])
# ---- Table A.4
sB = open(V + 'logs/04_gloss_and_macroF1_intervals.txt').read()
tt = {}
for line in sB.splitlines():
    m = re.match(r'(\S+) \| (\S+) \| (\S+)\s+3023\.0\s+718\.0\s+([\d.]+)\s+([\d.]+)', line)
    if m: tt[(m.group(1), m.group(2), m.group(3))] = (float(m.group(4)), float(m.group(5)))
# translation-type delta with CI: recompute from joined
j = pd.read_csv(V + 'predictions.csv', dtype={'Expression_ID': str, 'Participant_ID': str}, low_memory=False)
gl = pd.read_csv(V + 'gloss_flag.csv', dtype=str); gl = set(gl[gl.Gloss == 'True'].Expression_ID)
def delta(df_ids, s):
    m, md = SYS[s]
    q = j[(j.Model == m) & (j.Prompt_Mode == md) & j.Expression_ID.isin(df_ids)]
    w = q.pivot_table(index=['Expression_ID', 'Participant_ID'], columns='Representation', values='Correct', aggfunc='first').reset_index()
    w['d'] = w.English.astype(float) - w.Original.astype(float)
    gg = w.groupby('Participant_ID').d.agg(['sum', 'count']); r = np.random.default_rng(42)
    ix = r.integers(0, len(gg), size=(1000, len(gg))); v = gg['sum'].values[ix].sum(1) / gg['count'].values[ix].sum(1) * 100
    return len(w), 100 * w.Original.astype(float).mean(), 100 * w.English.astype(float).mean(), 100 * w.d.mean(), *np.quantile(v, [.025, .975])
allids = set(h.Expression_ID); trans = allids - gl
rows = [['System', 'All records: N', 'Accuracy, original / English (%)', 'Δₘ [95% CI]', 'Translation-type only: N', 'Accuracy, original / English (%)', 'Δₘ [95% CI]']]
order = sorted(SYS, key=lambda s: -dm.loc[s].Delta_m)
T['A4_order'] = order
for s in order:
    x = dm.loc[s]; n, ao, ae, d, lo, hi = delta(trans, s)
    rows.append([s, '3,741', f"{f1(x.A_orig)} / {f1(x.A_eng)}", f"{sg(x.Delta_m)} {ci(x.CI_low, x.CI_high)}", fn(n), f"{f1(ao)} / {f1(ae)}", f"{sg(d)} {ci(lo, hi)}"])
T['Table A.4'] = rows
# ---- Table A.5 from stageC (b) and (c)
sC = open(V + 'logs/05_robustness_checks.txt').read()
def parse(section):
    blk = sC.split(section)[1].split('\n\n')[0]
    out = {}
    for line in blk.strip().splitlines():
        m = re.match(r'(\S+) (\S+) \((\d+), np.float64\(([-\d.]+)\), np.float64\(([-\d.]+)\), np.float64\(([-\d.]+)\), np.float64\(([-\d.]+)\), np.float64\(([-\d.]+)\)\)', line)
        if m:
            s = next(k for k, v in SYS.items() if v == (m.group(1), m.group(2)))
            out[s] = [int(m.group(3))] + [float(m.group(i)) for i in range(4, 9)]
    return out
fx, up = parse('(b) Delta on reference-concordant records'), parse('(c) Delta on unique pairs')
rows = [['System', 'Fixed reference target: N', 'Δₘ [95% CI]', 'Unique pairs: N', 'Δₘ [95% CI]']]
for s in order:
    a, b = fx[s], up[s]
    rows.append([s, fn(a[0]), f"{sg(a[3])} {ci(a[4], a[5])}", fn(b[0]), f"{sg(b[3])} {ci(b[4], b[5])}"])
T['Table A.5'] = rows; T['fx'] = fx; T['up'] = up
# ---- Table A.6 / A.7
def sub(feat, val, s, r):
    x = t10[(t10.Feature == feat) & (t10.Value.astype(str) == str(val)) & (t10.Condition == cond(s, r))]
    return x.iloc[0]
langs = [('English', 'English'), ('Pidgin', 'Nigerian Pidgin'), ('Yoruba', 'Yoruba'), ('Yoruba exclamation', 'Yoruba exclamation'),
         ('Unclear', 'Unclear'), ('Pidgin/English', 'Pidgin/English'), ('Nigerian English', 'Nigerian English'),
         ('General Nigerian usage', 'General Nigerian usage'), ('Other/mixed (<25 each)', 'Other or mixed')]
five = ['RoBERTa', 'XLM-R', 'AfriSenti', 'Qwen zero-shot', 'Qwen cultural']
rows = [['Language (N; respondents)'] + [f"{s} O / E" for s in five]]
for v, lab in langs:
    x0 = sub('Lang', v, 'RoBERTa', 'Original')
    rows.append([f"{lab} ({fn(x0.N)}; {int(x0.Participants)})"] + [f"{f1(sub('Lang', v, s, 'Original').Accuracy)} / {f1(sub('Lang', v, s, 'English').Accuracy)}" for s in five])
T['Table A.6'] = rows
six = five + ['TextBlob']
rows = [['Subgroup (N; respondents)'] + [f"{s} O / E" for s in six], ['Referent'] + [''] * 6]
for v in ['Cultural lexicon', 'Self scenario', 'Other observation']:
    x0 = sub('Ref', v, 'RoBERTa', 'Original')
    rows.append([f"{v} ({fn(x0.N)}; {int(x0.Participants)})"] + [f"{f1(sub('Ref', v, s, 'Original').Accuracy)} / {f1(sub('Ref', v, s, 'English').Accuracy)}" for s in six])
rows.append(['Code-switching flag'] + [''] * 6)
for v, lab in [(0, 'Not flagged'), (1, 'Flagged')]:
    x0 = sub('Code_Switched', v, 'RoBERTa', 'Original')
    rows.append([f"{lab} ({fn(x0.N)}; {int(x0.Participants)})"] + [f"{f1(sub('Code_Switched', v, s, 'Original').Accuracy)} / {f1(sub('Code_Switched', v, s, 'English').Accuracy)}" for s in six])
T['Table A.7'] = rows
# ---- Table A.8
subs = ['All', 'Excl. priority-999', 'Excl. 46 corrupted originals', 'Excl. 64 mismatched English', 'Excl. all three', 'Unique pairs only']
rows = [['Condition', 'All (N = 3,741)', 'Excl. code 999 (3,427)', 'Excl. 46 corrupted (3,695)', 'Excl. 64 stale English (3,677)', 'Excl. all three (3,324)', 'Unique pairs (2,190)', 'Translation-type (3,023)', 'Gloss-type (718)']]
for s in SYS:
    for r, ab in [('Original', 'O'), ('English', 'E')]:
        c = cond(s, r); vals = [f1(t11[(t11.Condition == c) & (t11.Subset == k)].Accuracy.iloc[0]) for k in subs]
        tr_, gl_ = tt[(SYS[s][0], r, SYS[s][1])]
        rows.append([f"{s} {ab}"] + vals + [f1(tr_), f1(gl_)])
T['Table A.8'] = rows
json.dump(T, open(V + 'tables.json', 'w'), indent=0, default=float)
print('ok', T['ref_shift'], order)
