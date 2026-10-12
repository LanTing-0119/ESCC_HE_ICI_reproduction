"""Small, explicit input contracts. No private paths are stored in the package."""
from pathlib import Path
import os, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ['patient_id','cohort','treatment','image_included','age','sex','ECOG','TPS',
            'best_response','PFS_time','PFS_event','OS_time','OS_event']

def settings():
    cfg = json.loads((ROOT/'config.json').read_text())
    if os.environ.get('ESCC_CONFIG'):
        path=Path(os.environ['ESCC_CONFIG']).resolve()
        if path.is_relative_to(ROOT):
            raise ValueError('Keep the research configuration outside the public project.')
        cfg.update(json.loads(path.read_text()))
    cfg['mode'] = os.environ.get('ESCC_MODE',cfg['mode'])
    if cfg['mode'] not in ['template','demo','research']:
        raise ValueError('mode must be template, demo, or research')
    if cfg['mode']=='demo':
        cfg.update(clinical_csv='results/demo/work/clinical.csv',slide_manifest='results/demo/work/slides.csv',output_dir='results/demo')
    if cfg['mode']=='research':
        for key in ['clinical_csv','slide_manifest','output_dir']:
            if resolve(cfg[key]).is_relative_to(ROOT):
                raise ValueError(f'Research {key} must be outside the public directory.')
    return cfg

def resolve(path):
    p=Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT/p).resolve()

def destination(cfg):
    p=resolve(cfg['output_dir']);p.mkdir(parents=True,exist_ok=True);return p

def clinical(path):
    d=pd.read_csv(path,dtype={'patient_id':str,'cohort':str})
    missing=set(REQUIRED)-set(d)
    if missing:raise ValueError(f'Missing columns: {sorted(missing)}')
    if d.patient_id.isna().any() or d.patient_id.duplicated().any():
        raise ValueError('patient_id must be nonmissing and globally unique.')
    if not d.cohort.isin(['JUPITER06','ESCORT1ST']).all():raise ValueError('Unknown cohort')
    for col in ['treatment','image_included','PFS_event','OS_event']:
        if not d[col].isin([0,1]).all():raise ValueError(f'{col}: require 0 or 1')
    for col in ['PFS_time','OS_time']:
        if not np.isfinite(d[col]).all() or not d[col].gt(0).all():raise ValueError(f'{col}: positive months required')
    for col,lo,hi in [('age',18,120),('TPS',0,100),('ECOG',0,5)]:
        if not d[col].dropna().between(lo,hi).all():raise ValueError(f'{col} outside allowed range')
    if not d.sex.dropna().isin(['Male','Female']).all():raise ValueError('sex must be Male/Female or missing')
    d['best_response']=d.best_response.fillna('Missing')
    allowed=['CR','PR','SD','PD','NE','Non-CR/Non-PD','Missing']
    if not d.best_response.isin(allowed).all():raise ValueError('Unknown best_response category')
    return d

def inputs(cfg, require_analysis=True):
    d=clinical(resolve(cfg['clinical_csv']))
    if require_analysis and (len(d)<40 or d[d.image_included.eq(1)].groupby(['cohort','treatment']).size().reindex(
            pd.MultiIndex.from_product([['JUPITER06','ESCORT1ST'],[0,1]]),fill_value=0).min()<10):
        raise ValueError('The one-row template is readable but cannot support inference. Use optional demo mode or authorized complete data.')
    return d

def manifest(cfg,d):
    m=pd.read_csv(resolve(cfg['slide_manifest']),dtype={'patient_id':str,'slide_id':str})
    if m.slide_id.duplicated().any() or m.patient_id.isna().any():raise ValueError('Invalid slide mapping')
    ids=set(d.loc[d.image_included.eq(1),'patient_id'])
    if set(m.patient_id)!=ids:raise ValueError('Slide mapping must cover exactly the image-included patients')
    m['feature_path']=m.feature_path.map(lambda p:str(resolve(p)))
    if not m.n_patches.gt(0).all():raise ValueError('No patch features: run WSI extraction first')
    return m

def evaluation_frames(cfg,apparent=False,variant='full'):
    d=inputs(cfg);out=destination(cfg)/'work'
    jname=f'JUPITER06_{variant}_'+('apparent' if apparent else 'oof')+'.csv'
    paths={'JUPITER06':out/jname,'ESCORT1ST':out/f'ESCORT1ST_{variant}_external.csv'}
    frames={}
    for c,p in paths.items():
        s=pd.read_csv(p,dtype={'patient_id':str})
        base=d[d.cohort.eq(c)&d.image_included.eq(1)]
        if s.patient_id.duplicated().any() or set(s.patient_id)!=set(base.patient_id):raise ValueError('Incomplete/duplicate predictions')
        columns=['treatment','PFS_time','PFS_event','OS_time','OS_event']
        np.testing.assert_allclose(s.set_index('patient_id').loc[base.patient_id,columns],base[columns],rtol=0,atol=1e-10)
        if not np.isfinite(s.log_hr_z).all():raise ValueError('Nonfinite scores')
        if not (s.group==np.where(s.log_hr_z<0,'B','NB')).all():raise ValueError('Incorrect B/NB threshold')
        if c=='JUPITER06' and not apparent and set(s.fold)!=set(range(10)):raise ValueError('Require ten OOF folds')
        if c=='ESCORT1ST' and 'fold' in s:raise ValueError('External predictions must not contain fold strata')
        s=s.merge(base[['patient_id','TPS','best_response']],on='patient_id',validate='one_to_one')
        frames[c]=s
    return frames
