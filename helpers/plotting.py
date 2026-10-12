"""Publication-style figures from saved statistical tables."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

COLORS={('B',0):'#B3B7BC',('B',1):'#697480',('NB',0):'#0072B2',('NB',1):'#D55E00'}
NAMES={'JUPITER06':'JUPITER-06','ESCORT1ST':'ESCORT-1ST','Pooled':'Pooled'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})

def pvalue(p):return 'NE' if not np.isfinite(p) else '<.001' if p<.001 else f'{p:.3f}'.lstrip('0')
def interval(r,prefix=''):return 'NE' if not np.isfinite(r[prefix+'hr']) else f"{r[prefix+'hr']:.3f} ({r[prefix+'low']:.3f}–{r[prefix+'high']:.3f})"

def save(fig,out,stem,demo=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if demo:fig.text(.5,.015,'SIMULATED DATA — workflow demonstration only',ha='center',fontsize=11,color='#a33122',weight='bold')
    for ext in ['png','pdf','svg']:fig.savefig(out/f'{stem}.{ext}',dpi=180,facecolor='white')
    plt.close(fig)

def survival(out,stem,title,demo=False):
    from .survival_layout import draw_km,draw_forest
    out=Path(out)
    curves=pd.read_csv(out/f'{stem}_KM.csv');risk=pd.read_csv(out/f'{stem}_risk.csv')
    for d in [curves,risk]:d['group_key']=d.group+'_'+d.treatment.astype(str)
    hr=pd.read_csv(out/f'{stem}_HR.csv').rename(columns={'group':'subgroup','low':'ci_low','high':'ci_high'})
    inter=pd.read_csv(out/f'{stem}_interaction.csv').rename(columns={'ratio_hr':'ratio','ratio_low':'ci_low','ratio_high':'ci_high','ratio_p':'p'})
    inter['ratio_direction']='HR_B / HR_NB';inter['method']='Altman-Bland'
    data=dict(km_curves=curves,risk_table=risk,treatment_hr=hr,interaction=inter,survival_summary=pd.read_csv(out/f'{stem}_survival_summary.csv'))
    cfg=dict(colors={f'{g}_{t}':c for (g,t),c in COLORS.items()},
        linestyles={f'{g}_{t}':'-' if t else '--' for g in ['B','NB'] for t in [0,1]},
        group_labels={f'{g}_{t}':f'{g} | '+('ICI+CT' if t else 'CT') for g in ['B','NB'] for t in [0,1]},
        forest_colors={'Overall':'#697480','B':'#697480','NB':'#D55E00'},time_limit_months=48,time_ticks_months=[0,12,24,36,48])
    fig=plt.figure(figsize=(20,12),facecolor='white')
    for col,ep in enumerate(['PFS','OS']):draw_km(fig,ep,col,cfg,data);draw_forest(fig,ep,col,cfg,data)
    fig.text(.015,.986,title,fontsize=16,va='top',weight='bold')
    # Reserve a separate footer without altering the manuscript panel geometry.
    for ax in fig.axes:
        p=ax.get_position();ax.set_position([p.x0,.055+.93*p.y0,p.width,.93*p.height])
    for text in fig.texts:
        x,y=text.get_position();text.set_position((x,.055+.93*y))
    for artist in fig.artists:
        if hasattr(artist,'get_ydata'):artist.set_ydata(.055+.93*np.asarray(artist.get_ydata()))
    fig.text(.015,.041,'HR: ICI+CT / CT. Binary interaction: Altman-Bland. Internal OOF Cox: fold-stratified; external/apparent Cox: no fold strata. NR: not reached.',fontsize=9)
    save(fig,out,stem,demo)


def continuous(out,demo=False):
    out=Path(out);stats=pd.read_csv(out/'FigureS3_statistics.csv');curves=pd.read_csv(out/'FigureS3_curves.csv');hist=pd.read_csv(out/'FigureS3_histograms.csv')
    fig=plt.figure(figsize=(12.5,10.5));gs=fig.add_gridspec(4,2,height_ratios=[4,1,4,1],left=.09,right=.98,top=.89,bottom=.17,hspace=.6,wspace=.3)
    fig.suptitle('Figure S3 | Continuous treatment interactions',fontsize=17,y=.98)
    fig.text(.09,.935,'Raw predicted log HR(z); separate treatment × score Cox evaluation models',fontsize=10)
    for i,c in enumerate(['JUPITER06','ESCORT1ST']):
        for j,ep in enumerate(['PFS','OS']):
            ax=fig.add_subplot(gs[i*2,j]);hx=fig.add_subplot(gs[i*2+1,j],sharex=ax)
            d=curves[(curves.cohort==c)&(curves.endpoint==ep)];r=stats[(stats.cohort==c)&(stats.endpoint==ep)].iloc[0]
            for mask,color in [(d.log_hr_z<=0,'#697480'),(d.log_hr_z>=0,'#D55E00')]:
                a=d[mask];ax.fill_between(a.log_hr_z,a.low,a.high,color=color,alpha=.18);ax.plot(a.log_hr_z,a.hr,color=color)
            ax.axhline(1,color='#333',ls='--',lw=.8);ax.axvline(0,color='#555',ls=':',lw=.8);ax.set_yscale('log')
            ticks=[v for v in [.03125,.0625,.125,.25,.5,1,2,4,8,16,32,64] if ax.get_ylim()[0]<=v<=ax.get_ylim()[1]]
            ax.yaxis.set_major_locator(FixedLocator(ticks));ax.yaxis.set_major_formatter(FixedFormatter([f'{v:g}' for v in ticks]));ax.yaxis.set_minor_locator(NullLocator())
            ax.set_ylabel('Treatment HR (ICI+CT / CT)');ax.set_title(f'{chr(65+i*2+j)}  {NAMES[c]} | {ep}',loc='left',fontsize=11)
            ax.text(.03,.96,f'Interaction HR {interval(r)}\nP {pvalue(r.p)}',transform=ax.transAxes,va='top',fontsize=9)
            h=hist[hist.cohort.eq(c)];hx.bar((h.left+h.right)/2,h.n,width=(h.right-h.left)*.95,color='#697480')
            hx.set_xlabel('Predicted log HR(z)');hx.set_ylabel('Patients',fontsize=9);ax.tick_params(labelbottom=False)
    fig.text(.09,.06,'Pointwise robust 95% CIs. JUPITER: ten-fold OOF, stratified by fold. ESCORT: frozen external, no fold strata.\nOS evaluates the PFS-derived score; the interaction HR is per one raw score unit.',fontsize=9)
    save(fig,out,'FigureS3',demo)

def response(out,demo=False):
    out=Path(out);rates=pd.read_csv(out/'FigureS2_rates.csv');tests=pd.read_csv(out/'FigureS2_tests.csv');inter=pd.read_csv(out/'FigureS2_interactions.csv')
    rates=rates[rates.population.eq('all')];tests=tests[tests.population.eq('all')];inter=inter[inter.population.eq('all')]
    fig,axs=plt.subplots(2,2,figsize=(12.5,10));fig.subplots_adjust(left=.08,right=.98,top=.9,bottom=.13,hspace=.42,wspace=.24)
    fig.suptitle('Figure S2 | Objective response and disease control',fontsize=17,y=.97)
    for i,ep in enumerate(['ORR','DCR']):
        for j,c in enumerate(['JUPITER06','ESCORT1ST']):
            ax=axs[i,j]
            for k,g in enumerate(['B','NB']):
                for t,offset in [(1,-.19),(0,.19)]:
                    r=rates[(rates.cohort==c)&(rates.endpoint==ep)&(rates.group==g)&(rates.treatment==t)].iloc[0]
                    x=k+offset;ax.bar(x,r.rate*100,width=.35,color=COLORS[g,t],yerr=r.se*100,capsize=4,error_kw={'lw':1})
                    ax.text(x,r.rate*100+r.se*100+3,f'{int(r.positive)}/{int(r.n)}\n{100*r.rate:.1f}%',ha='center',fontsize=9)
                p=tests[(tests.cohort==c)&(tests.endpoint==ep)&(tests.group==g)].iloc[0].p
                ax.plot([k-.19,k-.19,k+.19,k+.19],[113,117,117,113],lw=.8,color='#333');ax.text(k,120,'P '+pvalue(p),ha='center',fontsize=9)
            p=inter[(inter.cohort==c)&(inter.endpoint==ep)].iloc[0].p
            ax.set_title(f'{chr(65+i*2+j)}  {NAMES[c]} | {ep}\nInteraction P {pvalue(p)}',loc='left',fontsize=12)
            ax.set_xticks([-.19,.19,.81,1.19],['ICI+CT\nB','CT\nB','ICI+CT\nNB','CT\nNB'])
            ax.set_ylim(0,130);ax.set_yticks([0,20,40,60,80,100]);ax.set_ylabel(ep+' (%)')
    fig.text(.08,.055,'Bars: proportions ± 1 binomial SE. Labels: responders/total patients. Within-group Fisher exact tests.\nInteraction: log(OR B / OR NB) Wald test. Missing/NE remain in primary denominators; evaluable sensitivity in CSV.',fontsize=9)
    save(fig,out,'FigureS2',demo)

def tps(out,demo=False):
    out=Path(out);counts=pd.read_csv(out/'FigureS5_counts.csv');tests=pd.read_csv(out/'FigureS5_tests.csv')
    fig=plt.figure(figsize=(16,8));a=fig.add_axes([.06,.30,.28,.53]);b=fig.add_axes([.42,.30,.55,.53])
    colors=['#477CAD','#CE8A45','#D3D7DD'];cats=['<1%','>=1%','Missing']
    for ax,specs in [(a,[(c,'Overall') for c in ['Pooled','JUPITER06','ESCORT1ST']]),(b,[(c,g) for c in ['Pooled','JUPITER06','ESCORT1ST'] for g in ['B','NB']])]:
        labels=[]
        for i,(c,g) in enumerate(specs):
            d=counts[(counts.cohort==c)&(counts.group==g)].set_index('category');bottom=0
            for cat,color in zip(cats,colors):
                r=d.loc[cat];ax.bar(i,r.percent,bottom=bottom,color=color,width=.65)
                if cat!='Missing':ax.text(i,bottom+r.percent/2,f'{int(r.n)}\n({r.percent:.1f}%)',ha='center',va='center',color='white',fontsize=10)
                bottom+=r.percent
            labels.append(NAMES[c]+('\n'+g if g!='Overall' else '')+f'\nn={int(d.total.iloc[0])}\nMissing: {int(d.loc["Missing","n"])}')
        ax.set_xticks(range(len(specs)),labels,fontsize=9);ax.tick_params(axis='x',length=0,pad=8);ax.set_ylim(0,126);ax.set_yticks([0,20,40,60,80,100]);ax.set_ylabel('Patients (%)');ax.spines['bottom'].set_visible(False)
    for c,x in [('JUPITER06',2.5),('ESCORT1ST',4.5)]:
        r=tests[tests.cohort.eq(c)].iloc[0];b.plot([x-.5,x-.5,x+.5,x+.5],[109,112,112,109],color='#333',lw=.8);b.text(x,116,'Pearson P '+pvalue(r.p),ha='center',fontsize=10)
    a.set_title('A  Overall distribution',loc='left');b.set_title('B  Distribution by benefit group',loc='left')
    fig.suptitle('Figure S5 | Distribution of PD-L1 TPS',fontsize=17,y=.95)
    fig.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(colors,['TPS < 1%','TPS ≥ 1%','Missing'])],loc='lower center',bbox_to_anchor=(.5,.13),ncol=3,frameon=False)
    fig.text(.06,.065,'Full-data model groups in both cohorts (JUPITER apparent; ESCORT frozen external). Nominal Pearson tests, no continuity or multiplicity correction.\nMissing TPS remains in bar denominators and is excluded from tests.',fontsize=9)
    save(fig,out,'FigureS5',demo)

def table(frame,out,stem,title,note='',demo=False):
    frame.to_csv(Path(out)/(stem+'_display.csv'),index=False)
    n=len(frame);width=max(10,len(frame.columns)*1.7);height=max(4,n*.44+2.1)
    fig,ax=plt.subplots(figsize=(width,height));ax.axis('off');fig.subplots_adjust(left=.025,right=.975,top=.84,bottom=.18)
    t=ax.table(cellText=frame.fillna('NE').values,colLabels=frame.columns,cellLoc='center',loc='center',bbox=[0,0,1,1]);t.auto_set_font_size(False);t.set_fontsize(9)
    for (i,j),cell in t.get_celld().items():
        cell.set_edgecolor('#d4d9de');cell.set_linewidth(.4);cell.set_facecolor('#dce6ed' if i==0 else '#f4f6f8' if i%2==0 else 'white')
    fig.suptitle(title,fontsize=16,y=.95);fig.text(.025,.07,note,fontsize=9,va='bottom')
    save(fig,out,stem,demo)
