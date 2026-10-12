"""Fixed-candidate Cox fitting and ten-fold training-only representations."""
import json, warnings
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from .io import inputs, manifest, destination
from .representation import fit_representation, map_all, reference_and_center

VARIANTS=['full','no_alpha','no_beta','gamma_only']
LABELS={'full':'Full','no_alpha':'No α','no_beta':'No βT','gamma_only':'γ only'}
CLIN=['patient_id','treatment','PFS_time','PFS_event','OS_time','OS_event']

def cox(d,ep,terms,penalty=0.,strata=None):
    cols=list(dict.fromkeys([ep+'_time',ep+'_event',*terms,*(strata or [])]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        f=CoxPHFitter(penalizer=penalty).fit(d[cols],ep+'_time',ep+'_event',robust=True,strata=strata or None)
    errors=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
    if errors:raise ValueError('; '.join(errors))
    if not np.isfinite(f.params_).all() or not np.isfinite(f.standard_errors_).all():raise ValueError('Nonfinite Cox estimates')
    return f

def fit_model(d,z,train_ids,reference,variant):
    base=d.set_index('patient_id');cols=[c for i,c in enumerate(z.columns) if i!=reference]
    parts=[base[['PFS_time','PFS_event']]]
    if variant in ['full','no_alpha']:parts.append(base[['treatment']])
    if variant in ['full','no_beta']:parts.append(z[cols])
    parts.append(z[cols].mul(base.treatment,axis=0).add_prefix('treatment_x_'))
    frame=pd.concat(parts,axis=1).loc[train_ids];terms=list(frame.columns[2:])
    with threadpool_limits(limits=1):f=cox(frame,'PFS',terms,penalty=.01)
    a=np.array([f.params_.get(c,0.) for c in z.columns]);g=np.array([f.params_.get('treatment_x_'+c,0.) for c in z.columns])
    model=dict(alpha=a.tolist(),gamma=g.tolist(),beta=float(f.params_.get('treatment',0)),reference=reference,
               n_parameters=len(terms),variant=variant,penalty=.01)
    return model,f

def predict(d,z,model):
    out=d[CLIN].copy();values=z.loc[out.patient_id].to_numpy()
    out['log_hr_z']=model['beta']+values@np.array(model['gamma'])
    out['group']=np.where(out.log_hr_z<0,'B','NB')
    out['risk_score']=values@np.array(model['alpha'])+out.treatment*out.log_hr_z
    out['n_parameters']=model['n_parameters'];return out

def interaction_covariance(d,z,reference,f):
    """Robust full-model covariance for exploratory spatial contrasts."""
    cols=[c for i,c in enumerate(z.columns) if i!=reference]
    base=d.set_index('patient_id')
    design=pd.concat([base[['PFS_time','PFS_event','treatment']],z[cols],
                      z[cols].mul(base.treatment,axis=0).add_prefix('treatment_x_')],axis=1)
    xx,tt,ee,ww,_,_,_=f._preprocess_dataframe(design)
    covariance=np.asarray(f._compute_sandwich_estimator((xx-f._norm_mean)/f._norm_std,tt,ee,ww))
    np.testing.assert_allclose(np.diag(covariance),f.standard_errors_.to_numpy()**2,rtol=1e-8,atol=1e-12)
    gamma_cov=np.zeros((len(z.columns),len(z.columns)))
    kept=[i for i in range(len(z.columns)) if i!=reference]
    positions=[list(f.params_.index).index('treatment_x_'+z.columns[i]) for i in kept]
    gamma_cov[np.ix_(kept,kept)]=covariance[np.ix_(positions,positions)]
    return gamma_cov

def run(cfg):
    if cfg['mode']=='research' and not cfg['enable_large_tasks']:
        raise ValueError('Set enable_large_tasks=true in the external research configuration to start full training.')
    d=inputs(cfg);m=manifest(cfg,d);out=destination(cfg)/'work';out.mkdir(exist_ok=True)
    j=d[d.image_included.eq(1)&d.cohort.eq('JUPITER06')].sort_values('patient_id').reset_index(drop=True)
    e=d[d.image_included.eq(1)&d.cohort.eq('ESCORT1ST')].sort_values('patient_id').reset_index(drop=True)
    jm=m[m.patient_id.isin(j.patient_id)];em=m[m.patient_id.isin(e.patient_id)]
    if cfg['pca_dimensions']!=128 or cfg['clusters']!=90 or cfg['folds']!=10 or cfg['ridge']!=.01 or cfg['patient_sampling_quantile']!=.5:
        raise ValueError('This package uses the fixed PCA128/K90/P50/10-fold/ridge=.01 candidate.')
    strata=j.treatment.astype(str)+'_'+j.PFS_event.astype(str)
    if strata.value_counts().min()<10:raise ValueError('Insufficient patients for treatment-by-PFS-event stratified ten-fold CV')
    split=StratifiedKFold(10,shuffle=True,random_state=cfg['seed'])
    folds=np.empty(len(j),int)
    for k,(_,va) in enumerate(split.split(j,strata)):folds[va]=k
    pd.DataFrame({'patient_id':j.patient_id,'fold':folds}).to_csv(out/'folds.csv',index=False)
    oof=[]
    for k in range(10):
        train=j[folds!=k];valid=j[folds==k];folder=out/f'fold_{k:02d}';folder.mkdir(exist_ok=True)
        pca,km=fit_representation(jm[jm.patient_id.isin(train.patient_id)],train,folder,cfg['seed'])
        prop,_=map_all(jm,pca,km,folder)
        z,center,reference=reference_and_center(prop,train.patient_id.tolist())
        z.to_csv(folder/'clr.csv');(folder/'representation.json').write_text(json.dumps(dict(reference=reference,center=center.tolist())))
        model,f=fit_model(j,z,train.patient_id.tolist(),reference,'full')
        pred=predict(valid,z,model);pred['fold']=k;oof.append(pred)
        (folder/'full_model.json').write_text(json.dumps(model,indent=2));f.summary.to_csv(folder/'full_coefficients.csv')
        print(f'Completed fold {k+1}/10',flush=True)
    pd.concat(oof).to_csv(out/'JUPITER06_full_oof.csv',index=False)
    folder=out/'full';folder.mkdir(exist_ok=True)
    pca,km=fit_representation(jm,j,folder,cfg['seed'])
    prop,_=map_all(jm,pca,km,folder);z,center,reference=reference_and_center(prop,j.patient_id.tolist())
    z.to_csv(folder/'JUPITER06_clr.csv');model,f=fit_model(j,z,j.patient_id.tolist(),reference,'full')
    model['clr_center']=center.tolist()
    model['gamma_covariance']=interaction_covariance(j,z,reference,f).tolist()
    (folder/'model.json').write_text(json.dumps(model,indent=2));f.summary.to_csv(folder/'coefficients.csv')
    # Store the robust coefficient covariance for exploratory interaction contrasts.
    import joblib
    joblib.dump(f,folder/'cox.joblib')
    predict(j,z,model).to_csv(out/'JUPITER06_full_apparent.csv',index=False)
    external_dir=folder/'external';external_dir.mkdir(exist_ok=True)
    ep,_=map_all(em,pca,km,external_dir)
    raw=np.log(ep);raw=raw.sub(raw.mean(axis=1),axis=0)
    ez=pd.DataFrame(raw.to_numpy()-center,index=ep.index,columns=z.columns)
    ez.to_csv(folder/'ESCORT1ST_clr.csv')
    predict(e,ez,model).to_csv(out/'ESCORT1ST_full_external.csv',index=False)
    return pd.concat(oof)

def ablate(cfg):
    """Refit the three deletions on the same stored fold representations."""
    d=inputs(cfg);out=destination(cfg)/'work'
    j=d[d.image_included.eq(1)&d.cohort.eq('JUPITER06')].sort_values('patient_id')
    e=d[d.image_included.eq(1)&d.cohort.eq('ESCORT1ST')].sort_values('patient_id')
    folds=pd.read_csv(out/'folds.csv',dtype={'patient_id':str}).set_index('patient_id').fold
    for variant in VARIANTS[1:]:
        frames=[]
        for k in range(10):
            folder=out/f'fold_{k:02d}';z=pd.read_csv(folder/'clr.csv',index_col=0)
            reference=json.loads((folder/'representation.json').read_text())['reference']
            train=folds.index[folds!=k].tolist();valid=j[j.patient_id.isin(folds.index[folds==k])]
            model,f=fit_model(j,z,train,reference,variant);pred=predict(valid,z,model);pred['fold']=k;frames.append(pred)
            (folder/f'{variant}_model.json').write_text(json.dumps(model,indent=2))
        pd.concat(frames).to_csv(out/f'JUPITER06_{variant}_oof.csv',index=False)
        z=pd.read_csv(out/'full/JUPITER06_clr.csv',index_col=0);ez=pd.read_csv(out/'full/ESCORT1ST_clr.csv',index_col=0)
        reference=json.loads((out/'full/model.json').read_text())['reference']
        model,f=fit_model(j,z,j.patient_id.tolist(),reference,variant)
        (out/'full'/f'{variant}_model.json').write_text(json.dumps(model,indent=2))
        predict(e,ez,model).to_csv(out/f'ESCORT1ST_{variant}_external.csv',index=False)
        print('Completed',variant,flush=True)
