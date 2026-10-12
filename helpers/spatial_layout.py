"""H&E, reference-coded regional maps and unique-covered-area summaries."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from shapely.geometry import box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from .plotting import save,NAMES

PALETTE={-1:'#2478ac',1:'#cc5946',0:'#babec3'}

def regional_paths(cells):
    """Merge same-class rectangles into editable vector outlines without filling holes."""
    for status,color in PALETTE.items():
        selected=[c for c in cells if c[-1]==status]
        if not selected:continue
        geometry=unary_union([box(x,y,x+w,y+h) for x,y,w,h,_ in selected])
        np.testing.assert_allclose(geometry.area,sum(c[2]*c[3] for c in selected),rtol=1e-10)
        polygons=[geometry] if geometry.geom_type=='Polygon' else list(geometry.geoms)
        paths=[]
        for polygon in polygons:
            polygon=orient(polygon,sign=1.)
            for ring in [polygon.exterior,*polygon.interiors]:
                points=np.asarray(ring.coords);codes=np.full(len(points),Path.LINETO,dtype=np.uint8)
                codes[0]=Path.MOVETO;codes[-1]=Path.CLOSEPOLY;paths.append(Path(points,codes))
        yield Path.make_compound_path(*paths),color

def case_panel(fig,box,info,demo):
    x,y,w,h=box
    title=info['patient_id']+' | Public TCGA ESCC' if demo else NAMES[info['cohort']]+' | '+info['group']+' / '+info['pfs_category']+' PFS'
    coverage='all segmented tissue patches' if info.get('complete_coverage') else 'sampled tissue patches'
    detail=f"{info['n_patches']:,} {coverage}; illustrative coefficients; no clinical B/NB assignment" if demo else f"Predicted HR(z)={info['hr']:.3g} | PFS={info['pfs']:.2f} months ("+('event)' if info['event'] else 'censored)')
    fig.text(x,y+h,title,fontsize=12,weight='bold',va='top');fig.text(x,y+h-.025,detail,fontsize=9,va='top',color='#40454c')
    cells=info['cells'];coords=np.array(cells)
    x0=coords[:,0].min();x1=(coords[:,0]+coords[:,2]).max();y0=coords[:,1].min();y1=(coords[:,1]+coords[:,3]).max()
    dx=(x1-x0)*.035;dy=(y1-y0)*.035;x0=max(0,x0-dx);x1=min(info['width'],x1+dx);y0=max(0,y0-dy);y1=min(info['height'],y1+dy)
    for j in [0,1]:
        ax=fig.add_axes([x+j*w*.29,y+.008,w*.275,h-.08])
        if j==0:ax.imshow(info['thumb'],extent=[0,info['width'],info['height'],0])
        else:
            for path,color in regional_paths(cells):ax.add_patch(PathPatch(path,facecolor=color,edgecolor='none',antialiased=False))
        ax.set(xlim=(x0,x1),ylim=(y1,y0),aspect='equal');ax.axis('off');ax.set_title('H&E' if j==0 else 'Regional map',fontsize=10)
        if np.isfinite(info['mpp']) and info['mpp']>0:
            length=1000/info['mpp']
            if length<(x1-x0)*.8:
                sx=x0+(x1-x0)*.04;sy=y1-(y1-y0)*.075
                ax.plot([sx,sx+length],[sy,sy],color='#1d2227',lw=1.6);ax.text(sx,sy-(y1-y0)*.035,'1 mm',fontsize=8)
    ax=fig.add_axes([x+w*.77,y+.025,w*.22,h-.11])
    values=np.array([info['areas'][status] for status in [1,-1,0]])*100
    ax.barh([2,1,0],values,color=[PALETTE[s] for s in [1,-1,0]],height=.48)
    ax.set(xlim=(0,124),ylim=(-.7,2.7),xticks=[0,50,100],yticks=[2,1,0],yticklabels=['Positive association','Negative association','Not significant'])
    ax.tick_params(axis='y',length=0,labelsize=9);ax.tick_params(axis='x',labelsize=8);ax.spines[['top','right','left']].set_visible(False)
    ax.set_title('Covered area (%)',fontsize=10);ax.set_axisbelow(True);ax.grid(axis='x',color='#eceff2',lw=.5)
    for yy,value in zip([2,1,0],values):ax.text(value+2,yy,'<0.1%' if 0<value<.1 else f'{value:.1f}%',fontsize=9,va='center')

def draw_cases(cases,out,demo,reference):
    if demo:
        n=len(cases);fig=plt.figure(figsize=(14,3*n+2),facecolor='white')
        fig.text(.03,.97,'Figure 4 | Public TCGA demonstration of the manuscript spatial layout',fontsize=17,weight='bold',va='top')
        fig.text(.03,.935,'H&E, standalone regional map and area percentages. Clinical case categories are unavailable for these public slides.',fontsize=11,va='top')
        rowheight=.77/n
        for i,case in enumerate(cases):case_panel(fig,[.035,.89-(i+1)*rowheight,.93,rowheight-.025],case,True)
    else:
        fig=plt.figure(figsize=(17.5,13.2),facecolor='white')
        for panel,title,top in [('A','B with short PFS; NB with long PFS',.967),('B','Comparison examples: B with long PFS; NB with short PFS',.505)]:
            fig.text(.029,top,panel+'  '+title,fontsize=16,weight='bold',va='top');fig.text(.97,top-.004,'ICI+CT',ha='right',va='top',fontsize=11)
            chosen=sorted([c for c in cases if c['panel']==panel],key=lambda c:(0 if c['cohort']=='JUPITER06' else 1,0 if c['group']=='NB' else 1))
            if len(chosen)>4:raise ValueError('Original Figure 4 layout supports at most four cases per panel')
            for i,case in enumerate(chosen):rr,cc=divmod(i,2);case_panel(fig,[.035+cc*.487,top-.239-rr*.204,.443,.195],case,False)
    fig.text(.035,.064,'Area denominator: unique patch-covered area; overlap counted once. White: uncovered. Gray: not significant / not estimable.',fontsize=9)
    fig.text(.035,.043,f'Directions are relative to C{reference:02d}. These are regional interaction associations, not local B/NB labels or verified treatment effects.',fontsize=9)
    if demo:fig.text(.035,.022,'Coefficients and uncertainty are artificially specified for visualization only. TCGA cluster identities do not match the paper model.',fontsize=9,color='#a33122')
    save(fig,out,'Figure4',False)
