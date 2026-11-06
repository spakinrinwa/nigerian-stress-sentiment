import os
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = lambda f: os.path.join(os.environ.get('DATA_DIR', os.path.join(_HERE, '..', 'data', 'restricted')), f)
OUTP = lambda f: os.path.join(os.environ.get('OUT_DIR', os.path.join(_HERE, '..', 'outputs')), f)
import pandas as pd, numpy as np, itertools, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import cohen_kappa_score, f1_score, precision_recall_fscore_support
OUT=OUTP('')
L=['Negative','Neutral','Positive']; R=1000; SEED=42
b=pd.read_csv(DATA('sentiment_benchmark.csv'),dtype={'Expression_ID':str,'Participant_ID':str},keep_default_na=False,encoding='utf-8-sig')
an=pd.read_csv(OUTP('Human_Sentiment_Annotation_3741_harmonised.csv'),dtype=str,keep_default_na=False,encoding='utf-8-sig')
def ref(p,q):
    p=p.strip().lower()
    if p in {"cultural_visible_stress_not_coping","cultural_mental_exhaustion_burnout"} or int(q)==400: return "Other observation"
    if p in {"cultural_exam_stress","cultural_study_fatigue_deadline_pressure"}: return "Self scenario"
    return "Cultural lexicon"
b['Ref']=[ref(p,q) for p,q in zip(b.Prompt_Source,b.Priority)]
lm={'english':'English','pidgin':'Pidgin','yoruba':'Yoruba','yoruba-exclamation':'Yoruba exclamation','unclear':'Unclear',
    'pidgin/english':'Pidgin/English','nigerian english':'Nigerian English','general nigerian usage':'General Nigerian usage'}
b['Lang']=b.Language.str.strip().str.lower().map(lm).fillna('Other/mixed (<25 each)')
b['Moj']=b.Original_Expression.str.contains('Ã|Â',regex=True)
mm=b[['Expression_ID','English_Interpretation']].merge(an[['Expression_ID','English_Interpretation']],on='Expression_ID')
b['EngMismatch']=b.Expression_ID.isin(mm[mm.iloc[:,1].str.strip()!=mm.iloc[:,2].str.strip()].Expression_ID)
b['Noise']=b.Priority==999
b['Dup']=b.duplicated(['Original_Expression','English_Interpretation'])
meta=b[['Expression_ID','Ref','Lang','Code_Switched','Construct','Moj','EngMismatch','Noise','Dup']]
j=pd.read_csv(OUTP('predictions.csv'),dtype={'Expression_ID':str,'Participant_ID':str},low_memory=False)
j=j.drop(columns=['Language','Construct','Referent_Type','Code_Switched']).merge(meta,on='Expression_ID',validate='many_to_one')
j['Cond']=j.Model+' | '+j.Representation+' | '+j.Prompt_Mode
rng=np.random.default_rng(SEED)
def cboot(df,stat,reps=R):
    g={k:v for k,v in df.groupby('Participant_ID')}; keys=np.array(list(g)); r=np.random.default_rng(SEED); out=[]
    for _ in range(reps):
        s=pd.concat([g[k] for k in r.choice(keys,len(keys))]); out.append(stat(s))
    return np.nanquantile(out,[.025,.975])
def fastboot(df,col):
    gg=df.groupby('Participant_ID')[col].agg(['sum','count']); r=np.random.default_rng(SEED)
    ids=r.integers(0,len(gg),size=(R,len(gg))); v=gg['sum'].values[ids].sum(1)/gg['count'].values[ids].sum(1)
    return np.quantile(v,[.025,.975])
# ---- T8 full metrics + coverage
den={'Classifier':3741,'Lexicon':3741,'ZeroShot':3741,'Cultural':3741}
rows=[]
for c,g in j.groupby('Cond'):
    t,p=g.Human_Label,g.Predicted_Sentiment; P,Rc,F,_=precision_recall_fscore_support(t,p,labels=L,zero_division=0)
    lo,hi=fastboot(g.assign(c=(t==p).astype(int)),'c')
    mf=cboot(g,lambda s:f1_score(s.Human_Label,s.Predicted_Sentiment,labels=L,average='macro',zero_division=0),300)
    rows.append(dict(Condition=c,Family=g.Model_Family.iloc[0],N=len(g),Participants=g.Participant_ID.nunique(),
      Coverage_pct=100*len(g)/den[g.Prompt_Mode.iloc[0]],Accuracy=100*(t==p).mean(),Acc_CI_low=100*lo,Acc_CI_high=100*hi,
      Macro_F1=f1_score(t,p,labels=L,average='macro',zero_division=0),MacroF1_CI_low=mf[0],MacroF1_CI_high=mf[1],
      Balanced_Acc=100*Rc.mean(),Kappa=cohen_kappa_score(t,p,labels=L),
      **{f'P_{l}':P[i] for i,l in enumerate(L)},**{f'R_{l}':Rc[i] for i,l in enumerate(L)},**{f'F1_{l}':F[i] for i,l in enumerate(L)}))
t8=pd.DataFrame(rows); t8.to_csv(OUT+'T8_performance_full.csv',index=False)
# ---- Delta_m paired (English - Original), matched IDs, cluster bootstrap; plus McNemar-ish counts
dm=[]
for (m,pm),g in j.groupby(['Model','Prompt_Mode']):
    o=g[g.Representation=='Original'][['Expression_ID','Participant_ID','Correct']]; e=g[g.Representation=='English'][['Expression_ID','Correct']]
    x=o.merge(e,on='Expression_ID',suffixes=('_O','_E')); x['d']=x.Correct_E.astype(int)-x.Correct_O.astype(int)
    gg=x.groupby('Participant_ID').agg(s=('d','sum'),n=('d','count')); r=np.random.default_rng(SEED); ids=r.integers(0,len(gg),size=(R,len(gg)))
    v=gg.s.values[ids].sum(1)/gg.n.values[ids].sum(1); lo,hi=np.quantile(v,[.025,.975])
    dm.append(dict(Model=m,Mode=pm,N_matched=len(x),Participants=x.Participant_ID.nunique(),A_orig=100*x.Correct_O.mean(),A_eng=100*x.Correct_E.mean(),
      Delta_m=100*x.d.mean(),CI_low=100*lo,CI_high=100*hi,Only_Eng_correct=int((x.d==1).sum()),Only_Orig_correct=int((x.d==-1).sum())))
pd.DataFrame(dm).to_csv(OUT+'T8b_delta_m.csv',index=False)
# ---- Experiment I: within-model shift (full overlap) + cross-model agreement
sh=[];tr=[]
for (m,pm),g in j.groupby(['Model','Prompt_Mode']):
    o=g[g.Representation=='Original'][['Expression_ID','Participant_ID','Predicted_Sentiment']]; e=g[g.Representation=='English'][['Expression_ID','Predicted_Sentiment']]
    x=o.merge(e,on='Expression_ID',suffixes=('_O','_E')); x['s']=(x.Predicted_Sentiment_O!=x.Predicted_Sentiment_E).astype(int)
    lo,hi=fastboot(x,'s')
    sh.append(dict(Model=m,Mode=pm,N_pairs=len(x),Participants=x.Participant_ID.nunique(),Shift_pct=100*x.s.mean(),CI_low=100*lo,CI_high=100*hi,
      Kappa=cohen_kappa_score(x.Predicted_Sentiment_O,x.Predicted_Sentiment_E,labels=L)))
    tr.append(pd.crosstab(x.Predicted_Sentiment_O,x.Predicted_Sentiment_E).assign(Model=m,Mode=pm))
pd.DataFrame(sh).to_csv(OUT+'T6_shifts.csv',index=False); pd.concat(tr).to_csv(OUT+'transitions.csv')
full=j
cm=[]
for rep in ['Original','English']:
    conds=sorted(full[full.Representation==rep].Cond.unique())
    for a,c in itertools.combinations(conds,2):
        x=full[full.Cond==a][['Expression_ID','Participant_ID','Predicted_Sentiment']].merge(full[full.Cond==c][['Expression_ID','Predicted_Sentiment']],on='Expression_ID')
        x['ag']=(x.iloc[:,2]==x.iloc[:,3]).astype(int); lo,hi=fastboot(x,'ag')
        cm.append(dict(Representation=rep,A=a,B=c,N=len(x),Agreement=100*x.ag.mean(),CI_low=100*lo,CI_high=100*hi,Kappa=cohen_kappa_score(x.iloc[:,2],x.iloc[:,3],labels=L)))
pd.DataFrame(cm).to_csv(OUT+'T7_cross_model.csv',index=False)
# ---- class distributions
dist=j.groupby('Cond').Predicted_Sentiment.value_counts(normalize=True).unstack().mul(100).round(1); dist['N']=j.groupby('Cond').size(); dist.to_csv(OUT+'T5_distributions.csv')
# ---- subgroups (n>=25)
sg=[]
for feat in ['Ref','Lang','Code_Switched']:
    for (c,v),g in j.groupby(['Cond',feat]):
        if len(g)>=25: sg.append(dict(Condition=c,Feature=feat,Value=v,N=len(g),Participants=g.Participant_ID.nunique(),Accuracy=100*g.Correct.mean(),
           Macro_F1=f1_score(g.Human_Label,g.Predicted_Sentiment,labels=L,average='macro',zero_division=0)))
pd.DataFrame(sg).to_csv(OUT+'T10_subgroups.csv',index=False)
# ---- sensitivity
se=[]
masks={'All':lambda d:d.index==d.index,'Excl. priority-999':lambda d:~d.Noise,'Excl. 46 corrupted originals':lambda d:~d.Moj,
       'Excl. 64 mismatched English':lambda d:~d.EngMismatch,'Unique pairs only':lambda d:~d.Dup,
       'Excl. all three':lambda d:~(d.Noise|d.Moj|d.EngMismatch)}
for c,g in j.groupby('Cond'):
    for k,f in masks.items():
        h=g[f(g)]; se.append(dict(Condition=c,Subset=k,N=len(h),Accuracy=100*h.Correct.mean(),Macro_F1=f1_score(h.Human_Label,h.Predicted_Sentiment,labels=L,average='macro',zero_division=0)))
pd.DataFrame(se).to_csv(OUT+'T11_sensitivity.csv',index=False)
# human agreement sensitivity
h=an.copy(); m={'negative':'Negative','neutral':'Neutral','positive':'Positive','po':'Positive'}
h['O']=h.Human_Original_Sentiment.str.strip().str.lower().map(m); h['E']=h.Human_English_Sentiment.str.strip().str.lower().map(m)
h=h.merge(meta,on='Expression_ID')
for k in ['All','Excl. 64 mismatched English','Excl. all three']:
    hh=h if k=='All' else (h[~h.EngMismatch] if 'mismatch' in k else h[~(h.Noise|h.Moj|h.EngMismatch)])
    print('HUMAN',k,len(hh),round(100*(hh.O==hh.E).mean(),1),round(cohen_kappa_score(hh.O,hh.E,labels=L),3))
print('done')
