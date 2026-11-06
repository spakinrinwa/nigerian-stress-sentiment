import os
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = lambda f: os.path.join(os.environ.get('DATA_DIR', os.path.join(_HERE, '..', 'data', 'restricted')), f)
OUTP = lambda f: os.path.join(os.environ.get('OUT_DIR', os.path.join(_HERE, '..', 'outputs')), f)
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score, cohen_kappa_score
L=['Negative','Neutral','Positive']
j=pd.read_csv(OUTP('predictions.csv'),dtype={'Expression_ID':str,'Participant_ID':str},low_memory=False)
b=pd.read_csv(DATA('sentiment_benchmark.csv'),dtype={'Expression_ID':str},keep_default_na=False,encoding='utf-8-sig')
an=pd.read_csv(OUTP('Human_Sentiment_Annotation_3741_harmonised.csv'),dtype={'Expression_ID':str},keep_default_na=False,encoding='utf-8-sig')
m={'negative':'Negative','neutral':'Neutral','positive':'Positive','po':'Positive'}
an['HO']=an.Human_Original_Sentiment.str.strip().str.lower().map(m); an['HE']=an.Human_English_Sentiment.str.strip().str.lower().map(m)
an['Conc']=an.HO==an.HE
b['RespGloss']=b.Original_Expression.str.contains(r' / |/ | - | = ',regex=True)
print('resp-gloss originals',b.RespGloss.sum(),'participants',b[b.RespGloss].Participant_ID.nunique())
j=j.merge(an[['Expression_ID','Conc']],on='Expression_ID').merge(b[['Expression_ID','RespGloss','Original_Expression','English_Interpretation']],on='Expression_ID')
# (a) majority baseline
for rep,col in [('Original','HO'),('English','HE')]:
    c=an[col].value_counts(); print('majority',rep,c.idxmax(),round(100*c.max()/len(an),1),'macroF1',round(f1_score(an[col],[c.idxmax()]*len(an),labels=L,average='macro',zero_division=0),3))
def dm(df):
    o=df[df.Representation=='Original'][['Expression_ID','Participant_ID','Correct']]; e=df[df.Representation=='English'][['Expression_ID','Correct']]
    x=o.merge(e,on='Expression_ID',suffixes=('_O','_E')); x['d']=x.Correct_E.astype(int)-x.Correct_O.astype(int)
    g=x.groupby('Participant_ID').agg(s=('d','sum'),n=('d','count')); r=np.random.default_rng(42); ids=r.integers(0,len(g),size=(1000,len(g)))
    v=g.s.values[ids].sum(1)/g.n.values[ids].sum(1)
    return len(x),round(100*x.Correct_O.mean(),1),round(100*x.Correct_E.mean(),1),round(100*x.d.mean(),1),round(100*np.quantile(v,.025),1),round(100*np.quantile(v,.975),1)
print('\n(b) Delta on reference-concordant records')
for (mo,pm),g in j.groupby(['Model','Prompt_Mode']): print(mo,pm,dm(g[g.Conc]))
print('\n(c) Delta on unique pairs (first occurrence)')
first=b.drop_duplicates(['Original_Expression','English_Interpretation']).Expression_ID
for (mo,pm),g in j.groupby(['Model','Prompt_Mode']): print(mo,pm,dm(g[g.Expression_ID.isin(first)]))
for (mo,pm),g in j[j.Representation=='Original'].groupby(['Model','Prompt_Mode']):
    print(mo,pm,'all',round(100*g.Correct.mean(),1),'no-gloss',round(100*g[~g.RespGloss].Correct.mean(),1),len(g[~g.RespGloss]),'gloss-only',round(100*g[g.RespGloss].Correct.mean(),1),len(g[g.RespGloss]))
print('\n(f) XLM-R minus AfriSenti, original accuracy, paired cluster bootstrap')
o=j[(j.Representation=='Original')]
x=o[o.Model=='Cardiff_Multilingual'][['Expression_ID','Participant_ID','Correct']].merge(o[o.Model=='AfriSenti_AfroXLMR'][['Expression_ID','Correct']],on='Expression_ID',suffixes=('_X','_A'))
x['d']=x.Correct_X.astype(int)-x.Correct_A.astype(int); g=x.groupby('Participant_ID').agg(s=('d','sum'),n=('d','count'))
r=np.random.default_rng(42); ids=r.integers(0,len(g),size=(1000,len(g))); v=g.s.values[ids].sum(1)/g.n.values[ids].sum(1)
print(round(100*x.d.mean(),1),np.round(100*np.quantile(v,[.025,.975]),1))
# per-referent negative share of reference (priming)
bb=b.copy()
def ref(p,q):
    p=p.strip().lower()
    if p in {'cultural_visible_stress_not_coping','cultural_mental_exhaustion_burnout'} or int(q)==400: return 'Other'
    if p in {'cultural_exam_stress','cultural_study_fatigue_deadline_pressure'}: return 'Self'
    return 'Lexicon'
bb['Ref']=[ref(p,q) for p,q in zip(bb.Prompt_Source,bb.Priority)]
k=bb.merge(an[['Expression_ID','HO']],on='Expression_ID')
print(pd.crosstab(k.Prompt_Source,k.HO,normalize='index').mul(100).round(1).sort_values('Negative').to_string())
