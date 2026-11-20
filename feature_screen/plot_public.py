"""Public aggregate redraws of supplementary plots, without patient records.
Original publication PDFs are preserved in figures/. Redraws use the same source
statistics and original canvas sizes but simplified styling in figures/public_reproduced/.
Exact original plotting cells require the authorized private replay inputs.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
COLORS=['#397B9C','#D57F75','#E1AE68','#A58AB9','#75B4A3','#98A5B3']
CATS=['EM','CNA','TMD','GWMD','GWFP','FLEN']
def family(f):
    if f.startswith('MOTIF_END'):return 'EM'
    if f.lower().startswith('flen'):return 'FLEN'
    return next(c for c in CATS if f.startswith(c))
def main(task_dir):
    d=Path(task_dir);out=d/'figures/public_reproduced';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'Arial' if 'Arial' in {x.name for x in font_manager.fontManager.ttflist} else 'DejaVu Sans','font.weight':'normal','axes.titleweight':'normal','axes.labelweight':'normal','pdf.fonttype':42})
    entries=json.loads((d/'figures/plot_manifest.json').read_text());generated=[];unavailable=[]
    for e in entries:
        data=d/'figures/data'/e['method']
        for name in e['names']:
            if name=='ClassifierScore_ByModel':
                unavailable.append(dict(name=name,reason='Individual-patient score strip plot requires withheld probability vectors; no aggregate substitute is labeled as the same plot'))
                for ext in ['pdf','png']:(out/f'{name}.{ext}').unlink(missing_ok=True)
                continue
            fig,ax=plt.subplots(figsize=e.get('canvas_inches',[9,7]));name_lower=name.lower()
            if 'auc_vs_featurecount' in name_lower or 'deltaauc' in name_lower or name=='FeatureCount-pink':
                x=pd.read_csv(data/'K_optimization.csv');x=x.rename(columns={'Num_Features':'k','Train_AUC':'train_auc','Validation_AUC':'validation_auc'})
                ax.plot(x.k,x.train_auc,label='Inner training',color=COLORS[0]);ax.plot(x.k,x.validation_auc,label='Inner validation',color=COLORS[1])
                ax.axvline(e['k'],ls='--',color='gray',label=f'K = {e["k"]}');ax.set(xlabel='Number of features',ylabel='ROC AUC',ylim=(0,1));ax.legend()
            elif 'proportion' in name_lower or 'biomarkertypes' in name_lower or 'top_feature' in name_lower:
                x=pd.read_csv(data/'selected_feature_scores.csv',header=None,names=['rank','feature','score']);x['family']=x.feature.map(family)
                if 'proportion' in name_lower:
                    counts=x.family.value_counts().reindex(CATS,fill_value=0);ax.bar(CATS,counts,color=COLORS);ax.set(ylabel='Number of selected features')
                else:
                    x=x.iloc[:30].iloc[::-1];ax.barh(x.feature,x.score,color=[COLORS[CATS.index(f)] for f in x.family]);ax.set(xlabel='Ranking score');ax.tick_params(axis='y',labelsize=8)
            elif 'confusion_matrix' in name_lower or 'per_class_accuracy_matrix' in name_lower:
                fname='cancer_only_confusion_normalized_all_true_cancer_patients.csv' if 'confusion_matrix' in name_lower else 'cancer_only_cross_indicator_accuracy_'+('Train' if 'train' in name_lower else 'Test')+'.csv'
                x=pd.read_csv(data/fname,index_col=0);im=ax.imshow(x.to_numpy(),vmin=0,vmax=1,cmap='OrRd');fig.colorbar(im,ax=ax)
                labels=['Breast','CRC','Gastric','Liver','Lung'];ax.set_xticks(range(5),labels,rotation=35,ha='right');ax.set_yticks(range(5),labels)
                for i in range(5):
                    for j in range(5):ax.text(j,i,f'{x.iloc[i,j]:.3f}',ha='center',va='center',fontsize=9)
                ax.set(xlabel='Predicted indicator',ylabel='True indicator')
            elif 'roc' in name_lower or name=='ReliefF' or name=='ReliefF_PR':
                is_pr=name.endswith('_PR');x=pd.read_csv(data/('PR_coordinates.csv' if is_pr else 'ROC_coordinates.csv'))
                if 'Dataset' in x:x=x[x.Dataset.eq('Test')]
                if 'six_class_ROC' in name:
                    v=pd.read_csv(d/'results/holdout/all_none_inner_validation_metrics.csv');v=v[v.method.eq(e['method'])]
                    best=sorted(v.to_dict('records'),key=lambda r:tuple(-round(r[k],12) for k in ['AUC','AUPRC','Recall','Specificity'])+(r['k'],r['model']))[0]['model'];x=x[x.Model.eq(best)]
                chosen=name.removesuffix('_ROC') if name.endswith('_ROC') else None
                if chosen is not None and chosen in set(x.Model):x=x[x.Model.eq(chosen)]
                assert len(x)>0, name
                for (model,cl),g in x.groupby(['Model','Class'],sort=False):
                    xx,yy=(g.Recall,g.Precision) if is_pr else (g.FPR,g.TPR)
                    ax.step(xx,yy,where='post',label=f'{model}, class {cl}',lw=1.4)
                if not is_pr:ax.plot([0,1],[0,1],'--',color='gray')
                ax.set(xlabel='Recall' if is_pr else 'False positive rate',ylabel='Precision' if is_pr else 'True positive rate',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=7,loc='best')
            elif 'AUC_dotplot' in name:
                x=pd.read_csv(data/'model_metrics_Test_and_apparent_training.csv')
                for ds,g in x.groupby('Dataset',sort=False):ax.scatter(g.AUC,g.Model,label=ds)
                ax.set(xlim=(0,1),xlabel='ROC AUC');ax.legend(fontsize=8)
            else:
                x=pd.read_csv(data/'model_metrics_Test_and_apparent_training.csv')
                dataset='Apparent final training' if 'train' in name_lower else 'Test';x=x[x.Dataset.eq(dataset)].set_index('Model')
                metrics=['AUC','AUPRC','Accuracy','Precision','Recall','F1','Specificity']
                if 'heatmap' in name_lower:
                    im=ax.imshow(x[metrics],vmin=0,vmax=1,cmap='OrRd',aspect='auto');fig.colorbar(im,ax=ax);ax.set_yticks(range(len(x)),x.index);ax.set_xticks(range(len(metrics)),metrics,rotation=35,ha='right')
                else:
                    for i,(model,row) in enumerate(x.iterrows()):ax.plot(metrics,row[metrics].to_numpy(dtype=float),'o-',label=model,color=COLORS[i])
                    ax.set(ylim=(0,1),ylabel=dataset+' score');ax.legend(fontsize=8)
            ax.set_title(name.replace('_',' '),fontsize=13,fontweight='normal');fig.tight_layout()
            for ext in ['pdf','png']:fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight')
            plt.close(fig);generated.append(name)
    (out/'manifest.json').write_text(json.dumps({'source':'../data; original canvas sizes in ../plot_manifest.json','scope':'Aggregate redraws; original manuscript PDFs remain in parent directory; exact original style replay uses authorized private predictions','generated':generated,'unavailable_without_private_data':unavailable},indent=2)+'\n')
    print(d.name,len(generated),'public supplementary aggregate redraws')
