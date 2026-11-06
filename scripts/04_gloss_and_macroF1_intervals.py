import os
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = lambda f: os.path.join(os.environ.get('DATA_DIR', os.path.join(_HERE, '..', 'data', 'restricted')), f)
OUTP = lambda f: os.path.join(os.environ.get('OUT_DIR', os.path.join(_HERE, '..', 'outputs')), f)
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score, cohen_kappa_score
L=['Negative','Neutral','Positive']
b=pd.read_csv(DATA('sentiment_benchmark.csv'),dtype={'Expression_ID':str},keep_default_na=False,encoding='utf-8-sig')
an=pd.read_csv(OUTP('Human_Sentiment_Annotation_3741_harmonised.csv'),dtype={'Expression_ID':str},keep_default_na=False,encoding='utf-8-sig')
e=an.English_Interpretation.str.strip()  # canonical English
GLOSS=r'(?i)^(an? |the )?(.{0,40})?(expression|exclamation|phrase|term|word|slang|idiom|interjection)\b|used (to|for|when|as)|often used|expression of|refers to|^no interpretable'
an['Gloss']=e.str.contains(GLOSS,regex=True)
print('gloss canonical',an.Gloss.sum(),'participants',an[an.Gloss].Participant_ID.nunique())
m={'negative':'Negative','neutral':'Neutral','positive':'Positive','po':'Positive'}
an['O']=an.Human_Original_Sentiment.str.strip().str.lower().map(m); an['E']=an.Human_English_Sentiment.str.strip().str.lower().map(m)
for g,d in an.groupby('Gloss'): print('HUMAN gloss=',g,len(d),round(100*(d.O==d.E).mean(),1),round(cohen_kappa_score(d.O,d.E,labels=L),3), d.E.value_counts().to_dict())
j=pd.read_csv(OUTP('predictions.csv'),dtype={'Expression_ID':str,'Participant_ID':str},low_memory=False).merge(an[['Expression_ID','Gloss']],on='Expression_ID')
j['Cond']=j.Model+' | '+j.Representation+' | '+j.Prompt_Mode
rows=[]
for (c,g),d in j.groupby(['Cond','Gloss']):
    rows.append(dict(Cond=c,Gloss=g,N=len(d),Acc=round(100*d.Correct.mean(),1),MF1=round(f1_score(d.Human_Label,d.Predicted_Sentiment,labels=L,average='macro',zero_division=0),3)))
r=pd.DataFrame(rows); print(r.pivot(index='Cond',columns='Gloss',values=['N','Acc','MF1']).to_string())
# Delta_m on translation-type only
dm=[]
for (mo,pm),g in j[~j.Gloss].groupby(['Model','Prompt_Mode']):
    o=g[g.Representation=='Original'][['Expression_ID','Participant_ID','Correct']]; en=g[g.Representation=='English'][['Expression_ID','Correct']]
    x=o.merge(en,on='Expression_ID',suffixes=('_O','_E')); x['d']=x.Correct_E.astype(int)-x.Correct_O.astype(int)
    gg=x.groupby('Participant_ID').agg(s=('d','sum'),n=('d','count')); rng=np.random.default_rng(42); ids=rng.integers(0,len(gg),size=(1000,len(gg)))
    v=gg.s.values[ids].sum(1)/gg.n.values[ids].sum(1)
    dm.append(dict(Model=mo,Mode=pm,N=len(x),Ao=round(100*x.Correct_O.mean(),1),Ae=round(100*x.Correct_E.mean(),1),Dm=round(100*x.d.mean(),1),lo=round(100*np.quantile(v,.025),1),hi=round(100*np.quantile(v,.975),1)))
print(pd.DataFrame(dm).to_string())
# Macro-F1 CI, 1000 reps, fast via per-participant confusion counts
out=[]
for c,g in j.groupby('Cond'):
    pids=g.Participant_ID.unique(); idx={p:i for i,p in enumerate(pids)}
    cm=np.zeros((len(pids),3,3)); li={l:i for i,l in enumerate(L)}
    for p,t,y in zip(g.Participant_ID,g.Human_Label,g.Predicted_Sentiment): cm[idx[p],li[t],li[y]]+=1
    rng=np.random.default_rng(42); f=[]
    for _ in range(1000):
        s=cm[rng.integers(0,len(pids),len(pids))].sum(0); tp=np.diag(s); P=np.divide(tp,s.sum(0),out=np.zeros(3),where=s.sum(0)>0); R=np.divide(tp,s.sum(1),out=np.zeros(3),where=s.sum(1)>0)
        F=np.divide(2*P*R,P+R,out=np.zeros(3),where=(P+R)>0); f.append(F.mean())
    out.append(dict(Cond=c,lo=round(np.quantile(f,.025),3),hi=round(np.quantile(f,.975),3)))
pd.DataFrame(out).to_csv('macroF1_CI_1000.csv',index=False); print(pd.DataFrame(out).to_string())
an[['Expression_ID','Gloss']].to_csv('gloss_flag.csv',index=False)
