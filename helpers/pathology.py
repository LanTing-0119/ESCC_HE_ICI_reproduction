"""Public TCGA layout demonstration and reusable spatial contrast calculations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import h5py,joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import norm
from .io import ROOT,resolve,destination,inputs
from .plotting import save

def patch_annotations(labels,gamma,gamma_cov):
    """Reference-coded interaction coefficients; no across-cluster recentering."""
    value=np.asarray(gamma);se=np.sqrt(np.maximum(0,np.diag(gamma_cov)));z=norm.ppf(.975)
    status=np.where(value+z*se<0,-1,np.where(value-z*se>0,1,0))
    return value[labels],se[labels],status[labels]

def union_area(coords,size,labels,gamma,gamma_cov):
    """Exact rectangular union partition for overlapping patch annotations.

    Each elementary covered rectangle gets the mean reference-coded coefficient of covering
    patches. Its covariance uses the cluster-frequency vector, including all
    coefficient covariances. Pixels outside sampled patch coverage are excluded.
    """
    k=len(gamma);xs=np.unique(np.r_[coords[:,0],coords[:,0]+size]);ys=np.unique(np.r_[coords[:,1],coords[:,1]+size])
    area={-1:0.,0:0.,1:0.};cells=[];critical=norm.ppf(.975)
    for x0,x1 in zip(xs[:-1],xs[1:]):
        ix=np.flatnonzero((coords[:,0]<=x0)&(coords[:,0]+size>=x1))
        if not len(ix):continue
        sub=coords[ix]
        for y0,y1 in zip(ys[:-1],ys[1:]):
            sel=ix[(sub[:,1]<=y0)&(sub[:,1]+size>=y1)]
            if not len(sel):continue
            weights=np.bincount(labels[sel],minlength=k)/len(sel)
            estimate=float(weights@gamma);se=float(np.sqrt(max(0,weights@gamma_cov@weights)))
            status=-1 if estimate+critical*se<0 else 1 if estimate-critical*se>0 else 0
            a=float((x1-x0)*(y1-y0));area[status]+=a
            cells.append((x0,y0,x1-x0,y1-y0,status))
    total=sum(area.values())
    return {s:a/total for s,a in area.items()},cells

def run(cfg):
    import openslide
    demo=cfg['mode']!='research'
    base=ROOT/'results/tcga_demo' if demo else destination(cfg)/'work/full'
    out=base/'figures' if demo else destination(cfg)/'figures_tables';out.mkdir(parents=True,exist_ok=True)
    if not demo and not cfg.get('pathology_manifest'):raise ValueError('Supply an external pathology_manifest listing the selected illustrative cases and coordinate paths.')
    manifest=pd.read_csv(base/'feature_manifest.csv' if demo else resolve(cfg['pathology_manifest']))
    if not demo and resolve(cfg['pathology_manifest']).is_relative_to(ROOT):raise ValueError('Keep research pathology inputs outside the public project')
    if len(manifest)>8 or len(manifest)<1:raise ValueError('Select 1–8 WSI cases for the spatial figure')
    arrays=[];coords=[];sizes=[]
    for r in manifest.itertuples():
        with h5py.File(resolve(r.feature_path),'r') as h:arrays.append(h['features'][:].astype('float32'))
        with h5py.File(resolve(r.coords_path),'r') as h:
            coords.append(h['coords'][:].astype(int));attrs=dict(h['coords'].attrs)
        if int(attrs.get('patch_size',0))!=256 or int(attrs.get('target_magnification',0))!=20 or int(attrs.get('overlap',-1))!=128:
            raise ValueError('Coordinate geometry does not match 20x / 256px / 50% overlap')
        sizes.append(int(attrs['patch_size_level0']))
    x=normalize(np.concatenate(arrays))
    if demo:
        pca=PCA(n_components=128,svd_solver='randomized',random_state=cfg['seed']).fit(x)
        z=pca.transform(x).astype('float32');km=MiniBatchKMeans(n_clusters=90,n_init=10,batch_size=8192,max_iter=300,random_state=cfg['seed']).fit(z)
        joblib.dump(pca,base/'demo_pca.joblib');joblib.dump(km,base/'demo_kmeans.joblib')
        # Deliberately illustrative coefficients: no TCGA clinical labels are used.
        reference=15;gamma=.45*np.sin(np.arange(90)*.51);gamma[reference]=0.
        cov=np.eye(90)*.08**2;cov[reference,:]=0.;cov[:,reference]=0.
        clinical=None
    else:
        pca=joblib.load(base/'pca.joblib');km=joblib.load(base/'kmeans.joblib');z=pca.transform(x).astype('float32')
        model=json.loads((base/'model.json').read_text());gamma=np.array(model['gamma']);cov=np.array(model['gamma_covariance']);reference=int(model['reference'])
        clinical=inputs(cfg).set_index('patient_id')
    labels=km.predict(z)
    coef=pd.DataFrame({'cluster':np.arange(90),'gamma':gamma,'se':np.sqrt(np.diag(cov)),'illustrative':demo})
    coef['low']=coef.gamma-norm.ppf(.975)*coef.se;coef['high']=coef.gamma+norm.ppf(.975)*coef.se
    coef=coef.sort_values(['gamma','cluster'],kind='stable').reset_index(drop=True)
    coef['plot_order']=np.arange(1,len(coef)+1)
    coef.to_csv(out/'FigureS4_coefficients.csv',index=False)
    counts=pd.read_csv(base/'patch_counts.csv').set_index('slide_id') if demo else None
    offset=0;area_rows=[];patches=[];cases=[]
    if not demo:
        if not {'panel','pfs_category'}.issubset(manifest.columns):raise ValueError('Research case manifest requires panel (A/B) and pfs_category (short/long); retain the authorized original case choices.')
        if not manifest.panel.isin(['A','B']).all() or not manifest.pfs_category.isin(['short','long']).all():raise ValueError('Invalid case panel/category')
    for i,r in enumerate(manifest.itertuples()):
        lab=labels[offset:offset+len(arrays[i])];offset+=len(arrays[i]);xy=coords[i];size=sizes[i]
        with openslide.OpenSlide(str(resolve(r.wsi_path))) as slide:
            thumb=np.array(slide.get_thumbnail((1200,1000)).convert('RGB'));w,h=slide.dimensions
            areas,cells=union_area(xy,size,lab,gamma,cov)
            area_rows.extend(dict(patient_id=r.patient_id,status=status,category={-1:'Negative association',0:'Not significant / not estimable',1:'Positive association'}[status],covered_area_fraction=v,reference_cluster=reference,illustrative=demo) for status,v in areas.items())
            info=dict(patient_id=r.patient_id,thumb=thumb,width=w,height=h,mpp=float(slide.properties.get('openslide.mpp-x',np.nan)),areas=areas,cells=cells,n_patches=len(xy))
            if demo:info['complete_coverage']=len(xy)==int(counts.loc[r.slide_id,'all_tissue_patches'])
            if not demo:
                patient=clinical.loc[r.patient_id]
                if patient.treatment!=1:raise ValueError('Original Figure 4 case design requires ICI+CT patients')
                scorefile=destination(cfg)/'work'/('JUPITER06_full_apparent.csv' if patient.cohort=='JUPITER06' else 'ESCORT1ST_full_external.csv')
                prediction=pd.read_csv(scorefile,dtype={'patient_id':str}).set_index('patient_id').loc[r.patient_id]
                group='B' if prediction.log_hr_z<0 else 'NB'
                expected=('short' if group=='B' else 'long') if r.panel=='A' else ('long' if group=='B' else 'short')
                if r.pfs_category!=expected:raise ValueError('Case panel does not match its B/NB and observed-PFS category')
                arm=clinical[clinical.cohort.eq(patient.cohort)&clinical.treatment.eq(1)&clinical.image_included.eq(1)]
                cut=arm.PFS_time.quantile(.25 if expected=='short' else .75)
                if (expected=='short' and (patient.PFS_time>cut or patient.PFS_event!=1)) or (expected=='long' and patient.PFS_time<cut):raise ValueError('Case does not satisfy the original within-cohort ICI-arm observed-PFS quartile rule')
                info.update(panel=r.panel,pfs_category=r.pfs_category,cohort=patient.cohort,group=group,hr=float(np.exp(prediction.log_hr_z)),pfs=float(patient.PFS_time),event=int(patient.PFS_event))
            cases.append(info)
            dist=km.transform(pca.transform(normalize(arrays[i])))
            for label in np.unique(lab):
                selected=np.flatnonzero(lab==label);idx=selected[np.argmin(dist[selected,label])]
                patches.append((dist[idx,label],int(label),np.array(slide.read_region(tuple(xy[idx]),0,(size,size)).convert('RGB').resize((128,128))),r.patient_id))
    pd.DataFrame(area_rows).to_csv(out/'Figure4_covered_area_fractions.csv',index=False)
    from .spatial_layout import draw_cases
    draw_cases(cases,out,demo,reference)
    chosen={}
    for dist,label,patch,patient in sorted(patches,key=lambda x:x[0]):
        if label not in chosen:chosen[label]=(patch,patient)
    selected_default=list(sorted(chosen))[:12] if demo else [int(r.cluster) for r in coef.itertuples() if r.low*r.high>0 and r.cluster in chosen]
    selection=cfg.get('atlas_clusters',selected_default)
    if not selection:selection=list(sorted(chosen))[:12]
    if any(k not in chosen for k in selection):raise ValueError('A requested atlas cluster has no patch in the supplied image manifest')
    selection=sorted(selection,key=lambda k:(gamma[k],k))
    draw_atlas(coef,chosen,selection,out,demo)
    return out

def draw_atlas(coef,chosen,selection,out,demo):
    """Draw the cached cluster atlas using Figure 4's interval-based colors."""
    from matplotlib.lines import Line2D
    from .spatial_layout import PALETTE
    coef=coef.sort_values(['gamma','cluster'],kind='stable').reset_index(drop=True).copy()
    coef['low']=coef.gamma-norm.ppf(.975)*coef.se
    coef['high']=coef.gamma+norm.ppf(.975)*coef.se
    coef['status']=np.where(coef.high<0,-1,np.where(coef.low>0,1,0))
    coef['color']=coef.status.map(PALETTE)
    coef['plot_order']=np.arange(1,len(coef)+1)
    coef.to_csv(out/'FigureS4_coefficients.csv',index=False)
    lookup=coef.set_index('cluster')
    selection=sorted(selection,key=lambda k:(lookup.loc[k,'gamma'],k))
    fig=plt.figure(figsize=(16,12));fig.suptitle('Figure S4 | Cluster atlas'+(' demonstration' if demo else ''),fontsize=18,y=.98)
    for j in range(2):
        ax=fig.add_axes([.06+j*.23,.13,.18,.76]);part=coef.iloc[j*45:(j+1)*45]
        for y,row in enumerate(part.itertuples()):
            ax.errorbar(row.gamma,y,xerr=norm.ppf(.975)*row.se,fmt='s',ms=2.5,color=row.color,lw=.7)
        ax.set_xlim(float(coef.low.min())-.05,float(coef.high.max())+.05)
        ax.set_yticks(range(len(part)),[f'C{k:02d}' for k in part.cluster],fontsize=7)
        for label,color in zip(ax.get_yticklabels(),part.color):label.set_color(color)
        ax.invert_yaxis();ax.axvline(0,color='#333',ls='--',lw=.8);ax.set_xlabel('Illustrative γ (interval)' if demo else 'Interaction γ (95% CI)')
    legend=[Line2D([0],[0],color=PALETTE[s],marker='s',lw=1,markersize=4,label=label) for s,label in [(-1,'Negative association: interval below 0'),(1,'Positive association: interval above 0'),(0,'Not significant / reference')]]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.955),ncol=3,frameon=False,fontsize=9)
    nrows=int(np.ceil(len(selection)/3));rowheight=.76/max(4,nrows)
    fig.set_size_inches(16,max(12,2.1*nrows+2))
    for j,label in enumerate(selection):
        color=lookup.loc[label,'color']
        ax=fig.add_axes([.55+(j%3)*.145,.9-(j//3+1)*rowheight,.13,rowheight*.82]);ax.imshow(chosen[label][0]);ax.axis('off');ax.set_title(f'C{label:02d}',fontsize=10,color=color)
        ax.add_patch(Rectangle((0,0),1,1,transform=ax.transAxes,fill=False,edgecolor=color,lw=2,clip_on=False))
    pd.DataFrame([{'cluster':k,'gamma':lookup.loc[k,'gamma'],'status':lookup.loc[k,'status'],'color':lookup.loc[k,'color'],'source_patient':chosen[k][1]} for k in selection]).to_csv(out/'FigureS4_representative_patches.csv',index=False)
    fig.text(.06,.045,('Illustrative coefficients and intervals; no TCGA clinical inference. Cluster identities differ from the paper-trained model.\n' if demo else 'Robust coefficient intervals from the full-data development model. Reference-cluster coefficient is fixed to zero.\n')+'Forest order: increasing gamma down the left column, then the right. Representative patches also follow increasing gamma.',fontsize=10)
    save(fig,out,'FigureS4',False);return out
