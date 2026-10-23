"""Recompute model results from the data and write results.json (run once; the dashboard only reads the JSON)."""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

NAME_MAP = {'Manchester City':'Man City','Manchester United':'Man United','Newcastle United':'Newcastle',
 'Tottenham Hotspur':'Tottenham','Wolverhampton Wanderers':'Wolves','Brighton & Hove Albion':'Brighton',
 'West Ham United':'West Ham','Nottingham Forest':"Nott'm Forest",'Leeds United':'Leeds','Chelsea FC':'Chelsea'}
CUTOFFS=[15,20,30]
FEATURES=['Points','GD','GF','GA','W','D','L','PPG','RecentFormPts5','EMA5']

def build_team_sequence(df, team, date_col='MatchDate'):
    rows=[]
    for _,r in df.iterrows():
        if r['HomeTeam']==team:
            gf,ga=r['FullTimeHomeGoals'],r['FullTimeAwayGoals']
            res=1 if r['FullTimeResult']=='H' else (0 if r['FullTimeResult']=='D' else -1)
        elif r['AwayTeam']==team:
            gf,ga=r['FullTimeAwayGoals'],r['FullTimeHomeGoals']
            res=1 if r['FullTimeResult']=='A' else (0 if r['FullTimeResult']=='D' else -1)
        else: continue
        rows.append({date_col:r[date_col],'GF':gf,'GA':ga,'Result':res})
    seq=pd.DataFrame(rows).sort_values(date_col).reset_index(drop=True)
    level,last,closes=0.0,None,[]
    for _,row in seq.iterrows():
        o=level
        if row['Result']==1: c=o+1.0; last='green'
        elif row['Result']==-1: c=o-1.0; last='red'
        else:
            if last=='green': c=max(o-o*0.5,0)
            elif last=='red': c=min(o+(0-o)*0.5,0)
            else: c=0.0; last='gray'
        closes.append(c); level=c
    seq['MomentumClose']=closes
    seq['EMA5']=seq['MomentumClose'].ewm(span=5,adjust=False).mean()
    seq['Points']=seq['Result'].map({1:3,0:1,-1:0})
    return seq

def feats_at(seq,cut):
    if len(seq)<cut: return None
    s=seq.iloc[:cut]; pts=s['Points'].sum(); gf=s['GF'].sum(); ga=s['GA'].sum()
    return {'Points':pts,'GD':gf-ga,'GF':gf,'GA':ga,'W':(s['Result']==1).sum(),'D':(s['Result']==0).sum(),
            'L':(s['Result']==-1).sum(),'PPG':pts/cut,'RecentFormPts5':s.tail(5)['Points'].sum(),'EMA5':s['EMA5'].iloc[-1]}

train=pd.read_csv('data/epl_final.csv'); train['MatchDate']=pd.to_datetime(train['MatchDate'])
rows={c:[] for c in CUTOFFS}; seasons=sorted(train['Season'].unique()); partial={}
for season in seasons:
    sdf=train[train['Season']==season]; teams=sorted(set(sdf.HomeTeam)|set(sdf.AwayTeam))
    seqs={t:build_team_sequence(sdf,t) for t in teams}
    partial[season]=int(min(len(s) for s in seqs.values()))
    tb=pd.DataFrame([(t,s['Points'].sum(),s['GF'].sum()-s['GA'].sum(),s['GF'].sum()) for t,s in seqs.items()],
        columns=['Team','Pts','GD','GF']).sort_values(['Pts','GD','GF'],ascending=False)
    top5=set(tb.head(5)['Team'])
    for t,seq in seqs.items():
        for c in CUTOFFS:
            f=feats_at(seq,c)
            if f: f.update(Season=season,Team=t,Top5=int(t in top5)); rows[c].append(f)
td={c:pd.DataFrame(rows[c]) for c in CUTOFFS}

def top5_by(df,keys): return set(df.sort_values(keys,ascending=False).head(5)['Team'])
out={'loso':{}, 'test':{}, 'meta':{}}
models={}
for c in CUTOFFS:
    df=td[c].copy(); oof=np.zeros(len(df)); n_ties=0; nv_pts=0; nv_gd=0
    for s in df.Season.unique():
        tr=df.Season!=s; te=df.Season==s
        sc=StandardScaler().fit(df.loc[tr,FEATURES]); clf=LogisticRegression(max_iter=1000).fit(sc.transform(df.loc[tr,FEATURES]),df.loc[tr,'Top5'])
        oof[te.values]=clf.predict_proba(sc.transform(df.loc[te,FEATURES]))[:,1]
        sd=df[te]; act=set(sd[sd.Top5==1].Team)
        nv_pts+=len(top5_by(sd,['Points'])&act); nv_gd+=len(top5_by(sd,['Points','GD'])&act)
    df['oof']=oof; mc=0
    for s in df.Season.unique():
        sd=df[df.Season==s]; mc+=len(top5_by(sd,['oof'])&set(sd[sd.Top5==1].Team))
    N=5*df.Season.nunique()
    out['loso'][str(c)]={'auc':round(roc_auc_score(df.Top5,oof),3),'model_hit':round(mc/N,3),'naive_hit':round(nv_pts/N,3),
        'naive_pts_gd_hit':round(nv_gd/N,3),'seasons':int(df.Season.nunique())}
    sc=StandardScaler().fit(df[FEATURES]); clf=LogisticRegression(max_iter=1000).fit(sc.transform(df[FEATURES]),df.Top5)
    models[c]=(sc,clf)
    out['loso'][str(c)]['coefs']={k:round(float(v),3) for k,v in zip(FEATURES,clf.coef_[0])}

test=pd.read_csv('data/epl_2025_26.csv'); test['HomeTeam']=test.HomeTeam.replace(NAME_MAP); test['AwayTeam']=test.AwayTeam.replace(NAME_MAP)
test=test.reset_index().rename(columns={'index':'MatchOrder'})
teams=sorted(set(test.HomeTeam)|set(test.AwayTeam)); seqs={t:build_team_sequence(test,t,'MatchOrder') for t in teams}
tb=pd.DataFrame([(t,s['Points'].sum(),s['GF'].sum()-s['GA'].sum(),s['GF'].sum()) for t,s in seqs.items()],columns=['Team','Pts','GD','GF']).sort_values(['Pts','GD','GF'],ascending=False)
ACTUAL=list(tb.head(5)['Team']); HARD={'Arsenal','Man City','Man United','Aston Villa','Liverpool'}
out['test']['actual_top5']=ACTUAL; out['test']['actual_matches_hardcoded_notebook']=set(ACTUAL)==HARD
out['test']['final_table_top7']=tb.head(7).values.tolist()
for c in CUTOFFS:
    fd=pd.DataFrame([dict(feats_at(seqs[t],c),Team=t) for t in teams])
    sc,clf=models[c]; fd['P']=clf.predict_proba(sc.transform(fd[FEATURES]))[:,1]
    pm=list(fd.sort_values('P',ascending=False).head(5)['Team'])
    npts=fd.sort_values('Points',ascending=False)
    pts_cut=npts.iloc[4]['Points']; tied=int((fd.Points==pts_cut).sum())
    nb=list(npts.head(5)['Team']); nb_gd=list(fd.sort_values(['Points','GD'],ascending=False).head(5)['Team'])
    A=set(ACTUAL)
    out['test'][str(c)]={'model':pm,'baseline':nb,'baseline_pts_gd':nb_gd,'model_correct':len(set(pm)&A),
      'baseline_correct':len(set(nb)&A),'baseline_pts_gd_correct':len(set(nb_gd)&A),'tied_at_5th_cutoff':tied,
      'model_misses':sorted(A-set(pm)),'model_false_alarms':sorted(set(pm)-A),
      'top8':fd.sort_values('P',ascending=False).head(8)[['Team','Points','GD','EMA5','P']].round(3).values.tolist()}
out['meta']={'partial_seasons':{k:v for k,v in partial.items() if v<38},'n_train_seasons':len(seasons)}
json.dump(out,open('data/results.json','w'),indent=1,default=str)
print(json.dumps(out,indent=1,default=str))
