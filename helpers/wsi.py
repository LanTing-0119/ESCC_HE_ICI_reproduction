"""TRIDENT segmentation, 20x patch coordinates and UNI-v2 feature extraction."""
from pathlib import Path
import os, json
import h5py
import numpy as np
import pandas as pd
from .io import ROOT

def run_tcga(cfg):
    import torch
    from trident import Processor
    from trident.patch_encoder_models import encoder_factory
    from trident.segmentation_models import segmentation_model_factory
    if not torch.cuda.is_available():raise RuntimeError('Use an allocated CUDA GPU for this UNI-v2 demonstration.')
    if not os.environ.get('UNI_V2_WEIGHTS'):raise ValueError('Set UNI_V2_WEIGHTS to your authorized local UNI-v2 checkpoint.')
    device=cfg.get('wsi_device','cuda:0');out=ROOT/'results/tcga_demo';out.mkdir(parents=True,exist_ok=True)
    slides=pd.read_csv(ROOT/'data/tcga_wsi/manifest.csv')
    processor=Processor(job_dir=str(out),wsi_source=str(ROOT/'data/tcga_wsi'),wsi_ext=['.svs'],reader_type='openslide',max_workers=2,skip_errors=False)
    # Optional in-process lookup override avoids changing the installed model registry.
    if os.environ.get('HEST_WEIGHTS'):
        import trident.segmentation_models.load as segload
        original=segload.get_weights_path
        segload.get_weights_path=lambda kind,name: os.environ['HEST_WEIGHTS'] if kind=='seg' and name=='hest' else original(kind,name)
        try:segmenter=segmentation_model_factory('hest')
        finally:segload.get_weights_path=original
    else:segmenter=segmentation_model_factory('hest')
    processor.run_segmentation_job(segmenter,seg_mag=10,batch_size=8,device=device)
    del segmenter;torch.cuda.empty_cache()
    processor.run_patching_job(target_magnification=20,patch_size=256,overlap=128,saveto='20x_256px_128px',visualize=True,min_tissue_proportion=0.)
    return extract_coordinates(cfg,processor,out,slides,device)

def extract_coordinates(cfg,processor,out,slides,device):
    """Extract embeddings from completed, geometry-checked tissue coordinates."""
    from trident.patch_encoder_models import encoder_factory
    limit=int(cfg.get('wsi_patch_limit',0));coords_name='full_coordinates' if limit<=0 else 'demo_coordinates'
    source=out/'20x_256px_128px/patches';limited=out/coords_name/'patches';limited.mkdir(parents=True,exist_ok=True)
    records=[]
    for row in slides.itertuples():
        path=source/f'{row.slide_id}_patches.h5'
        if not path.is_file():raise FileNotFoundError('Segmentation/patching did not complete for '+row.slide_id)
        with h5py.File(path,'r') as src:
            n=len(src['coords']);chosen=np.arange(n) if limit<=0 or n<=limit else np.sort(np.random.default_rng(cfg['seed']).choice(n,limit,replace=False))
            selected=src['coords'][:][chosen]
        existing=[p for p in (out/coords_name).rglob(row.slide_id+'.h5') if 'features' in str(p)]
        if existing:
            if not (limited/path.name).exists():raise ValueError('Feature cache lacks matching patch coordinates; archive results/tcga_demo and rerun.')
            with h5py.File(limited/path.name,'r') as cached:
                if not np.array_equal(cached['coords'][:],selected):raise ValueError('Patch selection changed. Archive results/tcga_demo before recomputing; existing features cannot be reused.')
        with h5py.File(path,'r') as src,h5py.File(limited/path.name,'w') as dst:
            for k,v in src.attrs.items():dst.attrs[k]=v
            for name,obj in src.items():
                if not isinstance(obj,h5py.Dataset):raise ValueError('Unexpected coordinate H5 group')
                values=obj[:][chosen] if len(obj.shape)>0 and obj.shape[0]==n else obj[()]
                new=dst.create_dataset(name,data=values)
                for k,v in obj.attrs.items():new.attrs[k]=v
            records.append(dict(slide_id=row.slide_id,all_tissue_patches=n,extracted_patches=len(chosen),patch_limit=limit))
    encoder=encoder_factory('uni_v2',weights_path=os.environ['UNI_V2_WEIGHTS'])
    processor.run_patch_feature_extraction_job(coords_dir=coords_name,patch_encoder=encoder,device=device,saveas='h5',batch_limit=16)
    rows=[]
    for row in slides.itertuples():
        candidates=[p for p in (out/coords_name).rglob(row.slide_id+'.h5') if 'features' in str(p)]
        if len(candidates)!=1:raise ValueError(f'Expected exactly one feature file for {row.slide_id}: {len(candidates)}')
        path=candidates[0]
        with h5py.File(path,'r') as h:
            shape=h['features'].shape
            expected=next(r['extracted_patches'] for r in records if r['slide_id']==row.slide_id)
            if len(shape)!=2 or shape!=(expected,1536) or not np.isfinite(h['features'][:]).all():raise ValueError('Invalid UNI-v2 output or mismatched coordinate cache')
        rows.append(dict(patient_id=row.patient_id,slide_id=row.slide_id,wsi_path=row.wsi_path,feature_path=str(path.relative_to(ROOT)),coords_path=str((limited/f'{row.slide_id}_patches.h5').relative_to(ROOT)),n_patches=shape[0]))
    result=pd.DataFrame(rows);result.to_csv(out/'feature_manifest.csv',index=False)
    pd.DataFrame(records).to_csv(out/'patch_counts.csv',index=False)
    # TRIDENT's operational metadata can contain local paths; retain it in the server run, not the public release.
    return result
