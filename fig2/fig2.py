"""Reproduce Figure 2 A-F from the tab-separated plotting inputs in this folder, without training or re-estimation.

Input TXT files and output PDFs share this directory, following the reference fig-folder layout.
Usage: python fig2.py [--output-dir PATH] [--font Arial]
"""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import argparse, json, platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
ROOT = Path(__file__).resolve().parent
DATA = ROOT
CANCERS = ['Lung', 'Breast', 'Colorectal', 'Liver', 'Gastric']
CLASSES = ['Healthy', 'Breast', 'CRC', 'Gastric', 'Liver', 'Lung']
CATEGORIES = ['EM', 'CNA', 'TMD', 'GWMD', 'GWFP', 'FLEN']
FEATURE_COLORS = ['#397B9C', '#D57F75', '#E1AE68', '#A58AB9', '#75B4A3', '#98A5B3']
ROC_COLORS = ['#8491B4', '#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F']
STEMS = {'A':'Figure2A_Feature_Composition','B':'Figure2B_Binary_Test_AUC',
         'C':'Figure2C_LOOCV_AUC','D':'Figure2D_Early_Stage_Sensitivity',
         'E':'Figure2E_Multiclass_ROC','F':'Figure2F_Confusion_Matrix'}

def configure(font='Arial'):
    available = {f.name for f in font_manager.fontManager.ttflist}
    selected = font if font in available else 'DejaVu Sans'
    plt.rcParams.update({'font.family':selected,'font.size':7.5,'font.weight':'normal',
        'axes.labelweight':'normal','axes.linewidth':.8,'axes.labelsize':7.5,
        'xtick.labelsize':7,'ytick.labelsize':7.5,'pdf.fonttype':42,'ps.fonttype':42,
        'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
    return selected

def read_table(name, **kwargs):
    return pd.read_csv(ROOT / name, sep='\t', **kwargs)

def read_inputs():
    a = read_table('feature_composition.txt').set_index('Cancer').loc[CANCERS].reset_index()
    b = read_table('binary_test_auc.txt').set_index('Cancer').loc[CANCERS].reset_index()
    cd = read_table('loocv_early_stage.txt')
    roc = read_table('multiclass_ROC.txt')
    auc = read_table('multiclass_ROC_AUC.txt').set_index('Class').loc[CLASSES]
    cm = read_table('confusion_counts.txt',index_col=0).loc[CLASSES,CLASSES]
    return {'A':a,'B':b,'CD':cd,'E':roc,'E_AUC':auc,'F':cm}

def axes_in(fig, cell, rect):
    x,y,w,h = cell; a,b,c,d = rect
    return fig.add_axes([x+w*a,y+h*b,w*c,h*d])

def mark_panel(fig, cell, panel):
    x,y,w,h = cell
    fig.text(x+.02*w,y+.965*h,f'({panel})',fontsize=10,ha='left',va='top')

# A. Feature-category composition: percentages are calculated from unchanged counts.
def panel_A(fig, cell, data):
    d=data['A']; ax=axes_in(fig,cell,[.27,.47,.57,.43]); y=np.arange(5)
    pct=d[CATEGORIES].to_numpy()/d.Selected_features.to_numpy()[:,None]*100
    left=np.zeros(5)
    for j,cat in enumerate(CATEGORIES):
        ax.barh(y,pct[:,j],left=left,height=.64,color=FEATURE_COLORS[j],edgecolor='white',linewidth=.4)
        left+=pct[:,j]
    for i in y:
        ax.text(pct[i,0]/2,i,f'{pct[i,0]:.1f}%',ha='center',va='center',fontsize=7.5,color='white')
        ax.text(1.025,i,f'n = {d.Selected_features.iloc[i]}',transform=ax.get_yaxis_transform(),va='center',fontsize=7)
    ax.set(ylim=(4.6,-.6),xlim=(0,100),xlabel='Percentage of selected features (%)')
    ax.set_yticks(y,CANCERS); ax.set_xticks([0,25,50,75,100]); ax.tick_params(axis='y',length=0,pad=6)
    for s in ['left','top','right']:ax.spines[s].set_visible(False)
    ax.xaxis.labelpad=2
    order=[0,3,1,4,2,5]; x,z,w,h=cell
    fig.legend(handles=[Patch(facecolor=FEATURE_COLORS[i],label=CATEGORIES[i]) for i in order],
        loc='lower left',bbox_to_anchor=(x+.30*w,z+.07*h),ncol=3,frameon=False,fontsize=7,
        handlelength=1,handletextpad=.4,columnspacing=1.1,labelspacing=.6,borderaxespad=0)
    return ax

# B. Original binary test AUCs, shown as the vertical lollipop used in the manuscript.
def panel_B(fig, cell, data):
    ax=axes_in(fig,cell,[.24,.39,.65,.52]); v=data['B'].Test_AUC.to_numpy(); x=np.arange(5)
    ax.vlines(x,.90,v,color='#9EB7C6',lw=1.4); ax.scatter(x,v,s=24,color='#193B58',zorder=3)
    ax.set(ylim=(.90,1.006),xlim=(-.6,4.6),ylabel='Test ROC AUC')
    ax.set_yticks([.90,.95,1],['0.90','0.95','1.00'])
    ax.set_xticks(x,CANCERS,rotation=42,ha='right',rotation_mode='anchor')
    for i,value in enumerate(v):ax.annotate(f'{value:.4f}',(i,value),xytext=(0,5),textcoords='offset points',ha='center',fontsize=7,color='#193B58')
    for s in ['top','right']:ax.spines[s].set_visible(False)
    return ax

# C. Frozen LOOCV AUC estimates and the original precomputed confidence intervals.
def panel_C(fig, cell, data):
    return forest(fig,cell,data,'C')

# D. Frozen stage I+II sensitivity at target 95% specificity (original achieved values in CSV).
def panel_D(fig, cell, data):
    return forest(fig,cell,data,'D')

def forest(fig,cell,data,panel):
    d=data['CD'].query('Panel == @panel').set_index('Cancer').loc[CANCERS].reset_index()
    v=d.Estimate.to_numpy(); y=np.arange(5); color='#397B9C' if panel=='C' else '#C98272'
    ax=axes_in(fig,cell,[.31 if panel=='C' else .35,.21,.43,.68])
    ax.errorbar(v,y,xerr=np.vstack([v-d.CI_lower.to_numpy(),d.CI_upper.to_numpy()-v]),
                fmt='o',ms=3.5,color=color,elinewidth=.65,capsize=1.7,capthick=.6)
    labels=CANCERS if panel=='C' else [f'{n} (n = {int(k)})' for n,k in zip(CANCERS,d.Sample_size)]
    ax.set_yticks(y,labels); ax.set_ylim(4.6,-.6); ax.tick_params(axis='y',length=0,pad=6)
    if panel=='C':
        ax.set_xlim(.75,1); ax.set_xticks([.75,.80,.85,.90,.95,1],['0.75','0.80','0.85','0.90','0.95','1.00']); ax.set_xlabel('ROC AUC',labelpad=2)
    else:
        ax.set_xlim(0,100); ax.set_xticks([0,20,40,60,80,100]); ax.set_xlabel('Sensitivity at 95% specificity (%)',labelpad=2,fontsize=7)
    ax.tick_params(axis='x',labelsize=6.5,length=3,pad=2)
    for i,value in enumerate(v):
        label=f'{value:.3f}' if panel=='C' else str(Decimal(str(value)).quantize(Decimal('.1'),rounding=ROUND_HALF_UP))+'%'
        ax.text(1.065,i,label,transform=ax.get_yaxis_transform(),va='center',fontsize=7.5)
    return ax

# E. Empirical six-class one-vs-rest ROC coordinates; no smoothing or model refitting.
def panel_E(fig, cell, data):
    ax=axes_in(fig,cell,[.30,.20,.47,.74])
    for name,color in zip(CLASSES,ROC_COLORS):
        d=data['E'][data['E'].Class.eq(name)]; value=data['E_AUC'].loc[name,'OvR_AUC']
        ax.plot(d.FPR,d.TPR,lw=.9,color=color,drawstyle='steps-post',label=f'{name} (AUC = {value:.4f})')
    ax.plot([0,1],[0,1],color='#a8adb2',lw=.55,ls=(0,(4,3)),zorder=0)
    ax.set(xlim=(-.015,1.015),ylim=(-.015,1.015),xlabel='False positive rate',ylabel='True positive rate')
    ax.set_xticks(np.linspace(0,1,6));ax.set_yticks(np.linspace(0,1,6));ax.set_box_aspect(1)
    ax.xaxis.labelpad=2;ax.yaxis.labelpad=2
    ax.legend(loc='lower right',frameon=False,fontsize=5.7,handlelength=1.7,labelspacing=.18,borderaxespad=.25,handletextpad=.5)
    return ax

# F. Original test confusion counts and percentages normalized WITHIN each true class.
def panel_F(fig, cell, data):
    ax=axes_in(fig,cell,[.29,.24,.50,.70]); counts=data['F'].to_numpy(); norm=counts/counts.sum(1,keepdims=True)
    cmap=LinearSegmentedColormap.from_list('paper_teal',['#f4f7f7','#d8e9e7','#9dc9c5','#549c9d','#286d7c','#194455'])
    for i in range(6):
        for j in range(6):
            # Explicit rectangles retain vector heatmap cells in PDF.
            ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=cmap(float(norm[i,j]))[:3],edgecolor='none',antialiased=False))
            ax.text(j,i,f'{counts[i,j]}\n{norm[i,j]:.1%}',ha='center',va='center',fontsize=5.8,linespacing=1.05,color='white' if norm[i,j]>.65 else '#202a30')
    ax.set(xlim=(-.5,5.5),ylim=(5.5,-.5),xlabel='Predicted class',ylabel='True class');ax.set_aspect('equal')
    ax.set_xticks(range(6),CLASSES,rotation=42,ha='right',rotation_mode='anchor');ax.set_yticks(range(6),CLASSES)
    ax.tick_params(length=0,pad=3);ax.xaxis.labelpad=1;ax.yaxis.labelpad=2
    cax=axes_in(fig,cell,[.81,.24,.018,.70]);cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap=cmap),cax=cax,ticks=[0,.25,.5,.75,1])
    cb.solids.set_rasterized(False);cb.solids.set_edgecolor("face");cb.set_label('Proportion within true class',fontsize=6.5,labelpad=3);cb.ax.tick_params(labelsize=6,length=2,pad=2)
    return ax

PANELS={'A':panel_A,'B':panel_B,'C':panel_C,'D':panel_D,'E':panel_E,'F':panel_F}

def make_single(panel,data=None):
    """Return one manuscript panel, using the same local layout as the combined PDF."""
    if data is None:data=read_inputs()
    fig=plt.figure(figsize=(93/25.4,56.667/25.4))
    PANELS[panel](fig,(0,0,1,1),data);mark_panel(fig,(0,0,1,1),panel)
    return fig

def validate_data(data):
    a=data['A']; assert np.array_equal(a[CATEGORIES].sum(1),a.Selected_features)
    selected=read_table('selected_features.txt')
    for cancer,row in a.set_index('Cancer').iterrows():
        n=selected[selected.Cancer.eq(cancer)].Category.value_counts()
        assert all(n.get(k,0)==row[k] for k in CATEGORIES)
    assert np.allclose(data['B'].Test_AUC,[.9747474747474748,.9614814814814816,.9833984375,.9922948771345272,.9782051282051282])
    cd=data['CD']; assert ((cd.CI_lower<=cd.Estimate)&(cd.Estimate<=cd.CI_upper)).all()
    assert np.allclose(cd[cd.Panel.eq('C')].Estimate,[.9077561669829224,.962962534355562,.9445810509231082,.929304600740349,.8725543120473996])
    assert np.allclose(cd[cd.Panel.eq('D')].Estimate,[56.25,80.625,600/7,2300/37,1200/23])
    counts=data['F'].to_numpy(); assert counts.sum()==294
    norm=counts/counts.sum(1,keepdims=True)
    original=read_table('confusion_row_normalized.txt',index_col=0)
    assert np.allclose(norm,original.to_numpy())
    assert np.array_equal(counts.sum(1),data['E_AUC'].Support.to_numpy())
    areas={}
    for name in CLASSES:
        d=data['E'][data['E'].Class.eq(name)];assert (np.diff(d.FPR)>=0).all() and (np.diff(d.TPR)>=0).all()
        area=float(np.trapezoid(d.TPR,d.FPR));assert np.isclose(area,data['E_AUC'].loc[name,'OvR_AUC'],atol=1e-12);areas[name]=area
    return dict(panel_order=list('ABCDEF'),feature_totals=a.Selected_features.tolist(),binary_test_auc=data['B'].Test_AUC.tolist(),roc_AUC_integral=areas,confusion_total=int(counts.sum()),all_data_checks_pass=True)

def save_pdf(fig,path):
    fig.canvas.draw()
    from matplotlib.text import Text
    renderer=fig.canvas.get_renderer();outside=[]
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text():
            bb=t.get_window_extent(renderer)
            if bb.x0<-.5 or bb.y0<-.5 or bb.x1>fig.bbox.width+.5 or bb.y1>fig.bbox.height+.5:outside.append(t.get_text())
    if outside:raise ValueError(f'Text outside page in {path.name}: {outside}')
    fig.savefig(path,format='pdf',facecolor='white',metadata={'Title':path.stem,'Subject':'Figure 2 reproducible vector artwork'})
    plt.close(fig)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT);parser.add_argument('--font',default='Arial')
    args=parser.parse_args(argv);args.output_dir.mkdir(parents=True,exist_ok=True)
    font=configure(args.font);data=read_inputs();checks=validate_data(data)
    for letter in 'ABCDEF':save_pdf(make_single(letter,data),args.output_dir/(STEMS[letter]+'.pdf'))
    fig=plt.figure(figsize=(186/25.4,170/25.4))
    for i,letter in enumerate('ABCDEF'):
        cell=((i%2)*.5,(2-i//2)/3,.5,1/3)
        PANELS[letter](fig,cell,data);mark_panel(fig,cell,letter)
    save_pdf(fig,args.output_dir/'Figure2_Combined.pdf')
    checks.update(font=font,requested_font=args.font,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,combined_size_mm=[186,170],outputs=7)
    print('Generated 6 panel PDFs and Figure2_Combined.pdf in',args.output_dir)
if __name__=='__main__':main()
