from pathlib import Path
import json
import numpy as np
from scipy.stats import rankdata
METHODS=['Fscore','ANOVA','ReliefF','Binomial']

CAPS={'lung':[60,60,150,60],'breast':[200,200,140,250],'crc':[200,200,200,150],'liver':[39,39,19,39],'gastric':[29,29,39,200],'pan6':[200,100,100,200],'pan_binary':[200,200,200,200]}

PARAMS=dict(booster='gbtree',max_depth=4,reg_lambda=3,reg_alpha=.2,subsample=.75,colsample_bytree=.75,min_child_weight=6,seed=0,nthread=2,gamma=.2,learning_rate=.05)

def dump(p,obj):
 tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)));tmp.replace(p)

def bootstrap_auc(y,probs,n=2000):
 rng=np.random.RandomState(42); classes=np.arange(probs.shape[1]);groups=[np.flatnonzero(y==c) for c in classes]
 ix=np.concatenate([rng.choice(g,size=(n,len(g)),replace=True) for g in groups],axis=1)
 yy=y[ix]; out={}
 for c in classes:
  positive=yy==c;s=probs[ix,c];r=rankdata(s,axis=1,method='average');np_=positive.sum(axis=1);nn=s.shape[1]-np_
  vals=((r*positive).sum(axis=1)-np_*(np_+1)/2)/(np_*nn)
  out[int(c)]=np.quantile(vals,[.025,.975]).tolist()
 return out

def balance(*args,**kwargs):
 raise RuntimeError('Sampling is disabled in the formal none analysis')
