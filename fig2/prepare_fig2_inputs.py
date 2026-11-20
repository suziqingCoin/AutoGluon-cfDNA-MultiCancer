"""Rebuild original-format TXT inputs from public formal aggregate tables."""
from pathlib import Path
import argparse,hashlib,json
import pandas as pd
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent
TASKS={'lung':'Lung_cancer','breast':'Breast_cancer','crc':'CRC_cancer','liver':'Liver_cancer','gastric':'Gastric_cancer'}
LABELS=dict(lung='Lung',breast='Breast',crc='Colorectal',liver='Liver',gastric='Gastric')
def main(output_dir=None):
    output_dir=Path(output_dir) if output_dir is not None else ROOT/"reproduced"
    output_dir.mkdir(parents=True,exist_ok=True)
    manifest=[]
    def read(p,**kw):
        manifest.append(dict(path=str(p.relative_to(REPO)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()));return pd.read_csv(p,**kw)
    def write(d,name):d.to_csv(output_dir/name,sep='\t',index=False)
    features=[];aa=[];bb=[]
    for task,folder in TASKS.items():
        f=read(REPO/folder/'results/final_selected_features.csv');f['Cancer']=LABELS[task];features.append(f.rename(columns={'feature':'Feature','family':'Category'}))
        aa.append(dict(Cancer=LABELS[task],Selected_features=len(f),**{c:int(f.family.eq(c).sum()) for c in ['EM','CNA','TMD','GWMD','GWFP','FLEN']}))
        b=read(REPO/folder/'results/tables/no_sampling_final_test_metrics_with_CI.csv').iloc[0];bb.append(dict(Cancer=LABELS[task],Test_AUC=b.AUC,AUC_CI_low=b.AUC_CI_low,AUC_CI_high=b.AUC_CI_high,method=b.method,k=b.k,model=b.model))
    write(pd.DataFrame(aa),'feature_composition.txt');write(pd.concat(features),'selected_features.txt');write(pd.DataFrame(bb),'binary_test_auc.txt')
    n=REPO/'nested_loocv/results'
    c=read(n/'Figure_2C.csv').rename(columns={'AUC':'Estimate','AUC_CI_low':'CI_lower','AUC_CI_high':'CI_upper','n':'Sample_size'});c['Panel']='C'
    d=read(n/'Figure_2D.csv');d=d[d.target_specificity.eq(.95)&d.stage.eq('I-II')].copy();d=d.rename(columns={'sensitivity':'Estimate','sensitivity_CI_low':'CI_lower','sensitivity_CI_high':'CI_upper','n_cases':'Sample_size','threshold':'Threshold','achieved_specificity':'Achieved_specificity'});d[['Estimate','CI_lower','CI_upper']]*=100;d['Panel']='D'
    cd=pd.concat([c,d],ignore_index=True);cd['Cancer']=cd.task.map(LABELS);write(cd,'loocv_early_stage.txt')
    base=REPO/'pancancer/figures/data'
    for src,dst in [('Figure2E_ROC_coordinates.csv','multiclass_ROC.txt'),('Figure2E_ROC_AUC.csv','multiclass_ROC_AUC.txt'),('Figure2F_confusion_counts.csv','confusion_counts.txt'),('Figure2F_confusion_row_normalized.csv','confusion_row_normalized.txt')]:write(read(base/src),dst)
    (output_dir/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Exported 8 TXT files from formal repository aggregates')
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'reproduced')
    main(parser.parse_args().output_dir)
