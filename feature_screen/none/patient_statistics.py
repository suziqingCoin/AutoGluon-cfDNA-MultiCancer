"""Patient-aligned bootstrap metrics, ties handled exactly as sklearn AP/AUC."""
import numpy as np
from scipy.stats import rankdata,beta
from sklearn.metrics import roc_auc_score,average_precision_score

def draws(y,n=2000):
 rng=np.random.RandomState(42);return np.concatenate([rng.choice(np.flatnonzero(y==c),size=(n,int((y==c).sum())),replace=True) for c in range(int(y.max())+1)],axis=1)
def average_precision_rows(positive,scores):
 order=np.argsort(-scores,axis=1,kind='stable');s=np.take_along_axis(scores,order,axis=1);a=np.take_along_axis(positive,order,axis=1)
 tp=np.cumsum(a,axis=1);end=np.c_[s[:,:-1]!=s[:,1:],np.ones(len(s),bool)];last=np.maximum.accumulate(np.where(end,tp,0),axis=1);previous=np.c_[np.zeros(len(s)),last[:,:-1]]
 return np.sum(np.where(end,(tp/np.arange(1,s.shape[1]+1))*(tp-previous),0),axis=1)/positive.sum(axis=1)
def boot(y,p,ix):
 nclass=p.shape[1];out=[];perclass={c:[] for c in range(nclass)}
 for start in range(0,len(ix),100):
  ind=ix[start:start+100];truth=y[ind];pred=p.argmax(axis=1)[ind];counts=[];aucs=[];aps=[];recalls=[];precisions=[];f1s=[];accs=[]
  for c in range(nclass):
   a=truth==c;b=pred==c;s=p[ind,c];r=rankdata(s,axis=1,method='average');np_=a.sum(axis=1);nn=s.shape[1]-np_;auc=((r*a).sum(axis=1)-np_*(np_+1)/2)/(np_*nn);ap=average_precision_rows(a,s);tp=(a&b).sum(axis=1);re=tp/np_;pr=np.divide(tp,b.sum(axis=1),out=np.zeros(len(ind),float),where=b.sum(axis=1)>0);f1=np.divide(2*pr*re,pr+re,out=np.zeros(len(ind),float),where=pr+re>0);acc=(a==b).mean(axis=1)
   counts.append(np_);aucs.append(auc);aps.append(ap);recalls.append(re);precisions.append(pr);f1s.append(f1);accs.append(acc)
   perclass[c].append(np.column_stack([auc,ap,re,pr,f1,acc]))
  counts=np.array(counts);cw=counts/counts.sum(axis=0);aucs=np.array(aucs);aps=np.array(aps);recalls=np.array(recalls)
  overall=np.column_stack([aucs[1] if nclass==2 else (aucs*cw).sum(axis=0),aps[1] if nclass==2 else (aps*cw).sum(axis=0),recalls[1] if nclass==2 else recalls.mean(axis=0),recalls[0],(truth==pred).mean(axis=1),np.array(precisions)[1] if nclass==2 else np.array(precisions).mean(axis=0),np.array(f1s)[1] if nclass==2 else np.array(f1s).mean(axis=0)])
  out.append(overall)
 return np.concatenate(out),{c:np.concatenate(v) for c,v in perclass.items()}
METRICS=['AUC','AUPRC','Recall','Specificity','Accuracy','Precision','F1'];CLASS_METRICS=['AUC','AUPRC','Recall','Precision','F1','OvR_Accuracy']
def exact_binomial(k,n):return (0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k)))
def validate():
 rng=np.random.RandomState(13)
 for nclass in [2,6]:
  y=np.arange(120)%nclass;p=rng.uniform(size=(120,nclass)).round(1);p/=p.sum(axis=1,keepdims=True);ind=draws(y,80);b,per=boot(y,p,ind)
  for j,k in enumerate(ind):
   for c in range(nclass):
    a=y[k]==c;assert np.isclose(per[c][j,0],roc_auc_score(a,p[k,c]));assert np.isclose(per[c][j,1],average_precision_score(a,p[k,c]))
 return dict(bootstrap_formula_matches_sklearn=True,n_binary_and_multiclass_replicates=160,includes_probability_ties=True)
if __name__=='__main__':print(validate())
