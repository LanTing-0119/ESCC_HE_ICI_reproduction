"""Independent simulated patients and patch features; no TCGA outcomes are invented."""
import numpy as np
import pandas as pd
import h5py
from .io import destination, ROOT
from .representation import _DEMO_FEATURE_CACHE

def generate(cfg):
    if cfg['mode']!='demo':raise ValueError('Synthetic generation requires explicit demo mode')
    out=destination(cfg)/'work';out.mkdir(exist_ok=True)
    features=out/'synthetic_features';features.mkdir(exist_ok=True)
    rng=np.random.default_rng(cfg['seed']);n=cfg['demo_patients_per_cohort'];patches=cfg['demo_patches_per_patient']
    prototypes=rng.normal(size=(90,1536)).astype('float32')
    prototypes/=np.linalg.norm(prototypes,axis=1,keepdims=True)
    rows=[];slides=[]
    for cohort,prefix in [('JUPITER06','SIM_DEV'),('ESCORT1ST','SIM_EXT')]:
        for i in range(n+30):
            pid=f'{prefix}_{i:04d}';t=i%2;included=int(i<n)
            latent=rng.normal();prognosis=rng.normal()
            # Deliberate effect heterogeneity for a runnable illustration, not trial estimates.
            loghr=-.35+1.1*latent
            pfs_true=rng.exponential(1/(.065*np.exp(.2*prognosis+t*loghr)))
            os_true=pfs_true+rng.exponential(12*np.exp(-.2*prognosis))
            censor=rng.uniform(27,55)
            response_prob=1/(1+np.exp(-(.1-t*loghr)))
            response=rng.choice(['CR','PR','SD','PD','NE','Missing'],p=[response_prob*.05,response_prob*.65,.18,.70-response_prob*.70,.07,.05])
            rows.append(dict(patient_id=pid,cohort=cohort,treatment=t,image_included=included,age=int(rng.integers(40,81)),
                sex=rng.choice(['Male','Female'],p=[.8,.2]),ECOG=int(rng.integers(0,2)),TPS=rng.choice([0.,5.,30.,np.nan],p=[.4,.35,.2,.05]),
                best_response=response,PFS_time=max(.05,min(pfs_true,censor)),PFS_event=int(pfs_true<=censor),
                OS_time=max(.1,min(os_true,censor)),OS_event=int(os_true<=censor)))
            if included:
                weights=np.exp(np.linspace(-1,1,90)*latent);weights/=weights.sum()
                labels=rng.choice(90,size=patches,p=weights)
                x=(prototypes[labels]+rng.normal(0,.018,size=(patches,1536))).astype('float32')
                path=features/f'{pid}.h5'
                if not path.exists():
                    with h5py.File(path,'w') as h:h.create_dataset('features',data=x)
                else:
                    with h5py.File(path,'r') as h:
                        if h['features'].shape!=x.shape or not np.array_equal(h['features'][:],x):
                            raise ValueError('Existing demo features differ: choose a new demo output directory.')
                _DEMO_FEATURE_CACHE[str(path)] = x
                slides.append(dict(patient_id=pid,slide_id=pid+'_SLIDE',wsi_path='',feature_path=str(path.relative_to(ROOT)),n_patches=patches))
    pd.DataFrame(rows).to_csv(out/'clinical.csv',index=False)
    pd.DataFrame(slides).to_csv(out/'slides.csv',index=False)
    return pd.DataFrame(rows)
