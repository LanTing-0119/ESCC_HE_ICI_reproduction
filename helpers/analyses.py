"""Small figure/table tasks shared by the interactive notebooks."""
import json
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency,mannwhitneyu
from . import statistics as st, plotting as plot
from .io import inputs, destination, evaluation_frames, manifest
from .training import VARIANTS,LABELS

def folder(cfg):
    out=destination(cfg)/'figures_tables';out.mkdir(exist_ok=True);return out

def survival(cfg):
    out=folder(cfg)
    for stem,c,apparent,title in [('Figure2','JUPITER06',False,'Figure 2 | JUPITER-06: ten-fold out-of-fold evaluation'),
            ('Figure3','ESCORT1ST',False,'Figure 3 | ESCORT-1ST: frozen external evaluation'),
            ('FigureS1','JUPITER06',True,'Figure S1 | JUPITER-06: apparent full-data performance')]:
        d=evaluation_frames(cfg,apparent)[c];curves=[];risk=[];hr=[];interaction=[];summaries=[]
        for ep in ['PFS','OS']:
            h,it=st.relative(d,ep,c=='JUPITER06' and not apparent);km,r=st.km_tables(d,ep)
            hr.append(h);interaction.append(it);curves.append(km);risk.append(r)
            summaries.append(st.survival_summary(d,ep))
        pd.concat(summaries).to_csv(out/f'{stem}_survival_summary.csv',index=False)
        for suffix,data in [('KM',pd.concat(curves)),('risk',pd.concat(risk)),('HR',pd.concat(hr)),('interaction',pd.DataFrame(interaction))]:data.to_csv(out/f'{stem}_{suffix}.csv',index=False)
        plot.survival(out,stem,title,cfg['mode']=='demo')
    return out

def absolute(cfg):
    out=folder(cfg);arms,contrasts=st.absolute(evaluation_frames(cfg))
    arms.to_csv(out/'absolute_arm_statistics.csv',index=False);contrasts.to_csv(out/'absolute_contrasts.csv',index=False)
    for metric,stem,title in [('rmst','Table1','Table 1 | Restricted mean survival time'),('survival','Table2','Table 2 | Fixed-time survival probabilities')]:
        rows=[];scale=100 if metric=='survival' else 1
        for r in contrasts[contrasts.metric.eq(metric)].itertuples():
            a=arms[(arms.cohort==r.cohort)&(arms.endpoint==r.endpoint)&(arms.group==r.group)]
            row={'Cohort':plot.NAMES[r.cohort],'Endpoint':f'{r.endpoint}, {r.tau} m','Group':r.group}
            for t,label in [(1,'ICI+CT'),(0,'CT')]:
                if r.group=='B minus NB':row[label]='—'
                else:
                    x=a[a.treatment.eq(t)].iloc[0];estimate=scale*x[metric];se=scale*np.sqrt(x[metric+'_var'])
                    row[label]=f'{estimate:.2f} ({estimate-st.Z*se:.2f}–{estimate+st.Z*se:.2f})'
            row['Difference\n(95% CI)']=f'{r.estimate:.2f} ({r.ci_low:.2f}–{r.ci_high:.2f})' if np.isfinite(r.estimate) else 'NE'
            row['P value']=plot.pvalue(r.p)+(' b' if r.group=='B minus NB' else ' a');rows.append(row)
        unit='months' if metric=='rmst' else 'percentage points (survival rates in %)'
        note=f'Units: {unit}. Treatment difference: ICI+CT minus CT.\na: within-group treatment-difference Wald test. b: B minus NB treatment-difference Wald test. All P values nominal.\nJUPITER uses OOF groups; ESCORT uses frozen external groups. Greenwood plug-in variances; predictions treated as fixed.'
        plot.table(pd.DataFrame(rows),out,stem,title,note,cfg['mode']=='demo')
    return contrasts

def response_tps(cfg):
    out=folder(cfg);rates,tests,interactions=st.response(evaluation_frames(cfg))
    for stem,data in [('FigureS2_rates',rates),('FigureS2_tests',tests),('FigureS2_interactions',interactions)]:data.to_csv(out/f'{stem}.csv',index=False)
    plot.response(out,cfg['mode']=='demo')
    counts,tests=st.tps(evaluation_frames(cfg,apparent=True))
    counts.to_csv(out/'FigureS5_counts.csv',index=False);tests.to_csv(out/'FigureS5_tests.csv',index=False)
    plot.tps(out,cfg['mode']=='demo');return out

def continuous(cfg):
    out=folder(cfg);stats,curves,hist=st.continuous(evaluation_frames(cfg))
    for name,data in [('statistics',stats),('curves',curves),('histograms',hist)]:data.to_csv(out/f'FigureS3_{name}.csv',index=False)
    plot.continuous(out,cfg['mode']=='demo');return stats

def clinical(cfg):
    out=folder(cfg);d=inputs(cfg);m=manifest(cfg,d)
    cols={f'{plot.NAMES[c]}\n{label}':d[d.cohort.eq(c)&d.image_included.eq(1)&d.treatment.eq(t)] for c in ['JUPITER06','ESCORT1ST'] for t,label in [(0,'CT'),(1,'ICI+CT')]}
    table=st.clinical_table(d,cols)
    selected=d[d.image_included.eq(1)]
    tests=st.clinical_comparisons(selected[selected.cohort.eq('JUPITER06')],selected[selected.cohort.eq('ESCORT1ST')])
    table['Cohort comparison\nP value']=[plot.pvalue(tests.get('age' if s.startswith('Age') else s.split(':')[0].replace('_category',''),np.nan)) if not s.endswith('Missing') else '—' for s in table.Characteristic]
    pd.DataFrame([tests]).to_csv(out/'TableS1_tests.csv',index=False)
    plot.table(table,out,'TableS1','Table S1 | Clinical characteristics by treatment arm','P values compare the two cohorts overall. Age: Mann–Whitney; categorical variables: Pearson, missing excluded.\nContinuous variables: median (IQR). Categorical percentages use all patients in the column. Repeated category P values are one omnibus test.',cfg['mode']=='demo')
    for cohort,stem in [('JUPITER06','TableS2'),('ESCORT1ST','TableS3')]:
        x=d[d.cohort.eq(cohort)];inc=x[x.image_included.eq(1)];exc=x[x.image_included.eq(0)]
        cols={'Full trial':x,'Image included':inc,'Not included':exc};table=st.clinical_table(x,cols)
        p=[]
        for label in table.Characteristic:
            if label=='Age, median (IQR)':
                pv=mannwhitneyu(inc.age.dropna(),exc.age.dropna(),alternative='two-sided',method='asymptotic').pvalue if len(exc) else np.nan
            elif label.startswith(('sex:','ECOG:','TPS:')) and not label.endswith('Missing'):
                col=label.split(':')[0];a=inc[col];b=exc[col]
                if col=='TPS':a=a.dropna().ge(1);b=b.dropna().ge(1)
                cats=sorted(set(a.dropna())|set(b.dropna()));cells=np.array([[int(v.eq(cat).sum()) for cat in cats] for v in [a,b]])
                try:pv=chi2_contingency(cells,correction=False).pvalue
                except ValueError:pv=np.nan
            else:pv=np.nan
            p.append(plot.pvalue(pv) if np.isfinite(pv) else '—')
        table['Included vs excluded\nP value']=p
        slide_n=int(m[m.patient_id.isin(inc.patient_id)].shape[0])
        slide_row={'Characteristic':'Mapped analysis slides','Full trial':str(slide_n),'Image included':str(slide_n),'Not included':'0','Included vs excluded\nP value':'—'}
        table=pd.concat([table.iloc[:1],pd.DataFrame([slide_row]),table.iloc[1:]],ignore_index=True)
        note=f'Included WSI: {slide_n}. Original trial counts require all randomized patients, including those without images.\nAge: two-sided Mann–Whitney test. Categorical variables: nominal Pearson test, missing excluded. Repeated category P values refer to one omnibus test.'
        plot.table(table,out,stem,f'{stem.replace("Table","Table ")} | {plot.NAMES[cohort]} inclusion comparison',note,cfg['mode']=='demo')
    return out

def ablation_table(cfg):
    out=folder(cfg);records=[];display=[]
    for c in ['JUPITER06','ESCORT1ST']:
        for variant in VARIANTS:
            d=evaluation_frames(cfg,variant=variant)[c]
            for ep in ['PFS','OS']:
                h,it=st.relative(d,ep,c=='JUPITER06');h=h.set_index('group')
                cont,_=st.cox_stat(d.assign(T_score=d.treatment*d.log_hr_z),ep,['treatment','log_hr_z','T_score'],'T_score',['fold'] if c=='JUPITER06' else None)
                row=dict(cohort=c,model=variant,n_parameters=int(d.n_parameters.iloc[0]),B_n=int(d.group.eq('B').sum()),NB_n=int(d.group.eq('NB').sum()),**it)
                for g in ['B','NB']:row.update({g+'_'+key:h.loc[g,key] for key in ['hr','low','high','p']})
                row.update({'continuous_'+k:v for k,v in cont.items()});records.append(row)
                display.append({'Cohort':plot.NAMES[c],'Model':LABELS[variant],'Endpoint':ep,'Terms':row['n_parameters'],'B / NB':f'{row["B_n"]} / {row["NB_n"]}',
                    'B treatment HR\n(95% CI)':plot.interval(row,'B_'),'NB treatment HR\n(95% CI)':plot.interval(row,'NB_'),
                    'HR B / HR NB\n(95% CI)':plot.interval(row,'ratio_'),'Binary P':plot.pvalue(row['ratio_p']),
                    'Continuous HR\n(95% CI)':plot.interval(row,'continuous_'),'Continuous P':plot.pvalue(cont['p'])})
    stats=pd.DataFrame(records);stats.to_csv(out/'TableS4_statistics.csv',index=False)
    note='All retained terms refitted to development PFS, ridge .01. Same candidate and fold representations; no candidate reselection.\nBinary: Altman–Bland. Continuous: treatment × raw log HR(z), adjusted for main effects. Internal Cox evaluation stratifies by fold.\nTerms are coefficient counts, not effective degrees of freedom. Nominal robust inference conditions on predictions; no between-model superiority test.'
    plot.table(pd.DataFrame(display),out,'TableS4','Table S4 | Fixed-candidate ablations',note,cfg['mode']=='demo')
    return stats
