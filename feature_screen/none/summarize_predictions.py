"""Recompute none Test metrics and conditional 2000-replicate CIs from authorized predictions."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from patient_statistics import draws,boot,METRICS,CLASS_METRICS,exact_binomial
from none_experiment import score

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args()
    d=pd.read_csv(a.predictions).sort_values('SampleID').reset_index(drop=True)
    assert d.SampleID.is_unique and d.strategy.eq('none').all()
    cols=sorted([c for c in d if c.startswith('p_')],key=lambda c:int(c[2:]));pr=d[cols].to_numpy();y=d.y_true.to_numpy()
    assert np.isfinite(pr).all() and np.allclose(pr.sum(1),1,atol=1e-5) and np.array_equal(d.y_pred,pr.argmax(1))
    b,per=boot(y,pr,draws(y));result=score(y,pr)
    for j,k in enumerate(METRICS):result[k+'_CI_low'],result[k+'_CI_high']=np.quantile(b[:,j],[.025,.975])
    n=int((y==0).sum());tn=int(((y==0)&d.y_pred.eq(0)).sum());lo,hi=exact_binomial(tn,n)
    result.update(healthy_n=n,healthy_TN=tn,Specificity_exact_CI_low=lo,Specificity_exact_CI_high=hi)
    classes=['Control','Breast cancer','CRC cancer','Gastric cancer','Liver cancer','Lung cancer'] if len(cols)==6 else ['Control',d.loc[d.y_true.eq(1),'Diagnosis'].iloc[0]]
    rows=[]
    for c,name in enumerate(classes):
        truth=y==c;pred=d.y_pred.to_numpy()==c;tp=int((truth&pred).sum());re=tp/truth.sum();prec=tp/pred.sum() if pred.sum() else 0
        r=dict(Class=name,support=int(truth.sum()),AUC=roc_auc_score(truth,pr[:,c]),AUPRC=average_precision_score(truth,pr[:,c]),Recall=re,Precision=prec,F1=2*prec*re/(prec+re) if prec+re else 0,OvR_Accuracy=float((truth==pred).mean()))
        for j,k in enumerate(CLASS_METRICS):r[k+'_CI_low'],r[k+'_CI_high']=np.quantile(per[c][:,j],[.025,.975])
        rows.append(r)
    a.output_dir.mkdir(parents=True,exist_ok=True);pd.DataFrame([result]).to_csv(a.output_dir/'overall_metrics.csv',index=False);pd.DataFrame(rows).to_csv(a.output_dir/'per_class_metrics.csv',index=False)
    print('Wrote aggregate Test metrics and conditional CIs; no patient records exported')
if __name__=='__main__':main()
