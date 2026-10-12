"""Conditional inference for fixed predictions; raw scores, nominal two-sided P values."""
import numpy as np
import pandas as pd
from scipy.stats import norm, fisher_exact, chi2_contingency, mannwhitneyu
from lifelines import KaplanMeierFitter
from lifelines.utils import qth_survival_times
from .training import cox
from .absolute import km_summary, wald
Z=norm.ppf(.975)

def cox_stat(d,ep,terms,target,strata=None):
    try:
        f=cox(d,ep,terms,strata=strata)
        b=float(f.params_[target]);s=float(f.standard_errors_[target])
        return dict(beta=b,se=s,hr=np.exp(b),low=np.exp(b-Z*s),high=np.exp(b+Z*s),p=2*norm.sf(abs(b/s)),status='SUCCESS',reason=''),f
    except Exception as exc:
        return dict(beta=np.nan,se=np.nan,hr=np.nan,low=np.nan,high=np.nan,p=np.nan,status='NOT_ESTIMABLE',reason=str(exc)),None

def relative(d,ep,stratified=False):
    strata=['fold'] if stratified else None;rows=[]
    for group in ['Overall','B','NB']:
        x=d if group=='Overall' else d[d.group.eq(group)]
        s,_=cox_stat(x,ep,['treatment'],'treatment',strata)
        s.update(group=group,endpoint=ep,n=len(x),events=int(x[ep+'_event'].sum()))
        for t,label in [(1,'ICI_CT'),(0,'CT')]:
            g=x[x.treatment.eq(t)];s[label+'_n']=len(g);s[label+'_events']=int(g[ep+'_event'].sum())
        rows.append(s)
    b,nb=rows[1:];delta=b['beta']-nb['beta'];se=np.hypot(b['se'],nb['se'])
    interaction=dict(endpoint=ep,ratio_hr=np.exp(delta),ratio_low=np.exp(delta-Z*se),ratio_high=np.exp(delta+Z*se),ratio_p=2*norm.sf(abs(delta/se)))
    return pd.DataFrame(rows),interaction

def km_tables(d,ep,ticks=(0,12,24,36,48)):
    curves=[];risks=[]
    for group in ['B','NB']:
        for t in [0,1]:
            x=d[d.group.eq(group)&d.treatment.eq(t)]
            if x.empty:continue
            f=KaplanMeierFitter().fit(x[ep+'_time'],x[ep+'_event'])
            for time,value in f.survival_function_.iloc[:,0].items():
                cens=int(f.event_table.loc[time,'censored'])
                curves.append(dict(endpoint=ep,group=group,treatment=t,time=time,survival=value,censored=cens))
            for time in ticks:risks.append(dict(endpoint=ep,group=group,treatment=t,time=time,at_risk=int(x[ep+'_time'].ge(time).sum()),cumulative_censored=int((x[ep+'_time'].lt(time)&x[ep+'_event'].eq(0)).sum())))
    return pd.DataFrame(curves),pd.DataFrame(risks)

def survival_summary(d,ep):
    rows=[]
    for group in ['Overall','B','NB']:
        x=d if group=='Overall' else d[d.group.eq(group)]
        for t in [1,0]:
            a=x[x.treatment.eq(t)]
            if a.empty:
                median=low=high=np.nan
            else:
                f=KaplanMeierFitter().fit(a[ep+'_time'],a[ep+'_event'])
                ci=qth_survival_times(.5,f.confidence_interval_).iloc[0]
                median=float(f.median_survival_time_);low=float(ci.iloc[0]);high=float(ci.iloc[1])
            rows.append(dict(endpoint=ep,subgroup=group,treatment=t,n=len(a),events=int(a[ep+'_event'].sum()),median=median,median_ci_low=low,median_ci_high=high))
    return pd.DataFrame(rows)

def absolute(frames):
    arms=[];contrasts=[]
    for cohort,d in frames.items():
        for ep,tau in [('PFS',12),('OS',24)]:
            stats={}
            for g in ['B','NB']:
                for t in [1,0]:
                    x=d[d.group.eq(g)&d.treatment.eq(t)]
                    if x.empty:raise ValueError('Empty B/NB treatment cell: absolute effects are not estimable')
                    s=km_summary(x[ep+'_time'],x[ep+'_event'],tau);stats[g,t]=s
                    arms.append(dict(cohort=cohort,endpoint=ep,tau=tau,group=g,treatment=t,**s))
            for metric in ['rmst','survival']:
                scale=100 if metric=='survival' else 1
                for group in ['B','NB','B minus NB']:
                    weights={(group,1):1,(group,0):-1} if group!='B minus NB' else {('B',1):1,('B',0):-1,('NB',1):-1,('NB',0):1}
                    r=wald(scale*sum(w*stats[k][metric] for k,w in weights.items()),scale**2*sum(w*w*stats[k][metric+'_var'] for k,w in weights.items()))
                    status='SUCCESS' if all(stats[k]['status']=='SUCCESS' for k in weights) else 'NOT_ESTIMABLE'
                    contrasts.append(dict(cohort=cohort,endpoint=ep,tau=tau,metric=metric,group=group,status=status,**r))
    return pd.DataFrame(arms),pd.DataFrame(contrasts)

def response(frames):
    rates=[];tests=[];interactions=[]
    for cohort,d in frames.items():
        for population in ['all','evaluable']:
            dd=d if population=='all' else d[~d.best_response.isin(['NE','Missing'])]
            for ep,positive in [('ORR',['CR','PR']),('DCR',['CR','PR','SD'])]:
                effects={}
                for g in ['B','NB']:
                    cells=[]
                    for t in [1,0]:
                        x=dd[dd.group.eq(g)&dd.treatment.eq(t)];n=len(x);k=int(x.best_response.isin(positive).sum())
                        p=k/n if n else np.nan
                        rates.append(dict(cohort=cohort,population=population,endpoint=ep,group=g,treatment=t,n=n,positive=k,rate=p,se=np.sqrt(p*(1-p)/n) if n else np.nan))
                        cells.append([k,n-k])
                    a=np.array(cells,float);p=fisher_exact(a.astype(int)).pvalue
                    # A zero cell makes the uncorrected log-OR interaction unestimable.
                    if (a>0).all():effects[g]=(np.log(a[0,0]*a[1,1]/(a[0,1]*a[1,0])),np.sum(1/a))
                    else:effects[g]=(np.nan,np.nan)
                    tests.append(dict(cohort=cohort,population=population,endpoint=ep,group=g,p=p))
                delta=effects['B'][0]-effects['NB'][0];se=np.sqrt(effects['B'][1]+effects['NB'][1])
                interactions.append(dict(cohort=cohort,population=population,endpoint=ep,or_ratio=np.exp(delta),p=2*norm.sf(abs(delta/se)),status='SUCCESS' if np.isfinite(se) else 'NOT_ESTIMABLE'))
    return pd.DataFrame(rates),pd.DataFrame(tests),pd.DataFrame(interactions)

def tps(frames):
    counts=[];tests=[];blocks={**frames,'Pooled':pd.concat(frames.values())}
    for cohort,d in blocks.items():
        for g in ['Overall','B','NB']:
            x=d if g=='Overall' else d[d.group.eq(g)]
            for cat,n in [('<1%',x.TPS.lt(1).sum()),('>=1%',x.TPS.ge(1).sum()),('Missing',x.TPS.isna().sum())]:
                counts.append(dict(cohort=cohort,group=g,category=cat,n=int(n),total=len(x),percent=100*n/len(x) if len(x) else np.nan))
        if cohort!='Pooled':
            cells=np.array([[int(d[d.group.eq(g)].TPS.ge(1).sum()),int(d[d.group.eq(g)].TPS.lt(1).sum())] for g in ['B','NB']])
            try:stat,p,df,expected=chi2_contingency(cells,correction=False);reason='';minimum=expected.min()
            except ValueError as exc:stat=p=minimum=np.nan;reason=str(exc)
            tests.append(dict(cohort=cohort,p=p,chi2=stat,min_expected=minimum,reason=reason))
    return pd.DataFrame(counts),pd.DataFrame(tests)

def continuous(frames):
    estimates=[];curves=[];hist=[]
    for cohort,d in frames.items():
        strata=['fold'] if cohort=='JUPITER06' else []
        grid=np.unique(np.r_[np.linspace(d.log_hr_z.quantile(.025),d.log_hr_z.quantile(.975),151),0.])
        count,edges=np.histogram(d.log_hr_z,bins=30)
        hist.extend(dict(cohort=cohort,left=a,right=b,n=int(n)) for n,a,b in zip(count,edges[:-1],edges[1:]))
        for ep in ['PFS','OS']:
            g=d.assign(T_score=d.treatment*d.log_hr_z);terms=['treatment','log_hr_z','T_score']
            s,f=cox_stat(g,ep,terms,'T_score',strata);estimates.append(dict(cohort=cohort,endpoint=ep,**s))
            if f is None:continue
            cols=[ep+'_time',ep+'_event',*terms,*strata]
            x,t,e,w,_,_,_=f._preprocess_dataframe(g[cols])
            cov=np.asarray(f._compute_sandwich_estimator((x-f._norm_mean)/f._norm_std,t,e,w))
            np.testing.assert_allclose(np.diag(cov),f.standard_errors_.to_numpy()**2,rtol=1e-8,atol=1e-12)
            names=list(f.params_.index);v=np.zeros((len(grid),len(names)))
            v[:,names.index('treatment')]=1;v[:,names.index('T_score')]=grid
            pred=v@f.params_.to_numpy();se=np.sqrt(np.einsum('ij,jk,ik->i',v,cov,v))
            curves.extend(dict(cohort=cohort,endpoint=ep,log_hr_z=z,hr=np.exp(b),low=np.exp(b-Z*s),high=np.exp(b+Z*s)) for z,b,s in zip(grid,pred,se))
    return pd.DataFrame(estimates),pd.DataFrame(curves),pd.DataFrame(hist)

def clinical_table(d,columns):
    """columns is an ordered mapping of displayed column names to patient frames."""
    rows=[]
    rows.append({'Characteristic':'Patients',**{k:str(len(x)) for k,x in columns.items()}})
    def value(x,col,cat):
        n=int(x[col].eq(cat).sum()) if cat!='Missing' else int(x[col].isna().sum())
        return f'{n} ({100*n/len(x):.1f}%)' if len(x) else 'NE'
    row={'Characteristic':'Age, median (IQR)'}
    for k,x in columns.items():
        q=x.age.quantile([.25,.5,.75]);row[k]=f'{q.loc[.5]:.1f} ({q.loc[.25]:.1f}–{q.loc[.75]:.1f})'
    rows.append(row)
    ecog_categories=sorted({0,1,*d.ECOG.dropna().astype(int).tolist()})+['Missing']
    for col,cats in [('sex',['Male','Female','Missing']),('ECOG',ecog_categories),('TPS_category',['<1%','>=1%','Missing'])]:
        for cat in cats:
            row={'Characteristic':f'{col.replace("_category","")}: {cat}'}
            for k,x in columns.items():
                xx=x.assign(TPS_category=np.where(x.TPS.isna(),None,np.where(x.TPS.lt(1),'<1%','>=1%')))
                row[k]=value(xx,col,cat)
            rows.append(row)
    for ep in ['PFS','OS']:rows.append({'Characteristic':ep+' events',**{k:f'{int(x[ep+"_event"].sum())} ({100*x[ep+"_event"].mean():.1f}%)' for k,x in columns.items()}})
    return pd.DataFrame(rows)

def clinical_comparisons(a,b):
    result={}
    result['age']=mannwhitneyu(a.age.dropna(),b.age.dropna(),alternative='two-sided',method='asymptotic').pvalue if len(a) and len(b) else np.nan
    for col in ['sex','ECOG','TPS']:
        aa=a[col].dropna();bb=b[col].dropna()
        if col=='TPS':aa=aa.ge(1);bb=bb.ge(1)
        cats=sorted(set(aa)|set(bb));cells=np.array([[int(v.eq(c).sum()) for c in cats] for v in [aa,bb]])
        try:result[col]=chi2_contingency(cells,correction=False).pvalue
        except ValueError:result[col]=np.nan
    return result
