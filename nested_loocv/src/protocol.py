import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS']:
 os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import json,hashlib,random
from functools import lru_cache
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,recall_score
ROOT=Path(os.environ.get('CFDNA_NESTED_OUTPUT',Path(__file__).resolve().parents[1])).resolve()
SOURCE_ROOT=Path(__file__).resolve().parents[1]
RAW=Path(os.environ.get('CFDNA_RAW',SOURCE_ROOT.parent/'data/cohort.csv'))
DIAG={'lung':'Lung cancer','breast':'Breast cancer','crc':'CRC cancer','liver':'Liver cancer','gastric':'Gastric cancer'}
COUNTS={'lung':1686,'breast':1773,'crc':1709,'liver':1672,'gastric':1648}
METHODS=['Fscore','ANOVA','ReliefF','Binomial']
CAPS={'lung':[60,60,150,60],'breast':[200,200,140,250],'crc':[200,200,200,150],'liver':[39,39,19,39],'gastric':[29,29,39,200]}
PARAMS=dict(booster='gbtree',max_depth=4,reg_lambda=3,reg_alpha=.2,subsample=.75,colsample_bytree=.75,min_child_weight=6,seed=0,nthread=1,gamma=.2,learning_rate=.05,objective='binary:logistic',eval_metric='auc')
MODELS={'LightGBM','RandomForest','XGBoost','NeuralNetFastAI','WeightedEnsemble_L2'}
def dump(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x)));tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@lru_cache(maxsize=1)
def raw_sha():return sha(RAW)
def signature():
 return hashlib.sha256(''.join(sha(p) for p in sorted((SOURCE_ROOT/'src').glob('*.py')) if p.name!='report.py').encode()).hexdigest()
def seed():
 import torch
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(1)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass

@lru_cache(maxsize=5)
def load(task):
 a=pd.read_csv(RAW);features=a.columns[4:].tolist();assert a.SampleID.is_unique and a.SampleID.notna().all();assert np.isfinite(a[features].to_numpy()).all()
 d=a[a.Diagnosis.isin(['Control',DIAG[task]])].reset_index(drop=True)
 assert len(d)==COUNTS[task] and (d.Label==0).sum()==1550
 assert np.array_equal(d.Label.to_numpy(),d.Diagnosis.ne('Control').astype(int).to_numpy())
 return d,features

def score(y,p):
 pred=p>.5 # argmax([1-p,p]) assigns exact ties to Control
 return dict(AUC=float(roc_auc_score(y,p)),AUPRC=float(average_precision_score(y,p)),Recall=float(recall_score(y,pred,zero_division=0)),Specificity=float((~pred[y==0]).mean()))
def choose(rows):return sorted(rows,key=lambda r:tuple(-round(r[k],12) for k in ['AUC','AUPRC','Recall','Specificity'])+(r['k'],METHODS.index(r['method']),r['model']))[0]
def optimize(X,y,V,yv,order,cap,path):
 import xgboost as xgb
 rows=[];best=-np.inf;bestk=1;over=decline=0;prev=0
 for k in range(1,min(cap,len(order))+1):
  cols=order[:k];dt=xgb.DMatrix(X[:,cols],label=y);dv=xgb.DMatrix(V[:,cols],label=yv)
  m=xgb.train(PARAMS,dt,num_boost_round=50,evals=[(dt,'train'),(dv,'eval')],early_stopping_rounds=10,verbose_eval=False)
  at=roc_auc_score(y,m.predict(dt));av=roc_auc_score(yv,m.predict(dv));rows.append(dict(k=k,train_auc=at,validation_auc=av,best_iteration=m.best_iteration))
  if av>best:best=float(av);bestk=k
  over=over+1 if at-av>.1 else 0;decline=decline+1 if av<prev else 0;prev=av
  if over>=10 or decline>=5:break
 pd.DataFrame(rows).to_csv(path,index=False);return bestk

def family(f):
 if f.startswith('MOTIF_END'):return 'EM'
 if f.lower().startswith('flen'):return 'FLEN'
 for c in ['CNA','TMD','GWMD','GWFP']:
  if f.startswith(c):return c
 raise ValueError(f)
