"""Historical four rankings, fitted only on the inner training patients.
Vectorized ReliefF scoring preserves skrebate 0.8.3 distances/neighbors and formulas.
"""
import itertools
import numpy as np
from scipy.stats import binomtest,chi2
from sklearn.feature_selection import f_classif
from skrebate import ReliefF

class VectorReliefF(ReliefF):
 def _run_algorithm(self):
  assert np.isfinite(self._X).all(), 'Use upstream skrebate for missing data'
  assert self._class_type in ('binary','multiclass')
  X=self._X; y=self._y; n,p=X.shape
  continuous=np.array([self.attr[h][0]=='continuous' for h in self._headers])
  ranges=np.array([self.attr[h][3] if continuous[j] else 1 for j,h in enumerate(self._headers)])
  std=np.array([self.attr[h][4] if continuous[j] else 0 for j,h in enumerate(self._headers)])
  scores=np.zeros(p)
  for i in range(n):
   nn=self._find_neighbors(i); hits=y[nn]==y[i]; diff=np.abs(X[nn]-X[i])
   diff[:,continuous]/=ranges[continuous]
   if self.data_type=='mixed':diff[:,continuous]=np.where(np.abs(X[nn][:,continuous]-X[i,continuous])>std[continuous],1.,diff[:,continuous])
   diff[:,~continuous]=(X[nn][:,~continuous]!=X[i,~continuous])
   s=-diff[hits].mean(axis=0) if hits.any() else np.zeros(p)
   if self._class_type=='binary':
    if (~hits).any():s+=diff[~hits].mean(axis=0)
   else:
    misses=int((~hits).sum())
    if misses:
     for c in self.mcmap:
      if c!=y[i]:
       m=y[nn]==c;s+=diff[m].sum(axis=0)*int(m.sum())*(len(self.mcmap)-1)/(misses**2)
   scores+=s/n
  return scores

def rank(X,y,method):
 a=np.asarray(X,dtype=float); labels=np.unique(y)
 if method=='Fscore':
  means=np.array([a[y==c].mean(axis=0) for c in labels]); variances=np.array([a[y==c].var(axis=0,ddof=1) for c in labels])
  scores=((means-a.mean(axis=0))**2).sum(axis=0)/np.maximum(variances.sum(axis=0),1e-300)
 elif method=='ANOVA':scores=f_classif(a,y)[0]
 elif method=='ReliefF':
  scores=VectorReliefF(n_neighbors=10,n_jobs=1).fit(a,y).feature_importances_
 elif method=='Binomial':
  # Cache exact two-sided binomial p-values for each (training-only) group size.
  sizes={int((y==c).sum()) for c in labels}; table={n:np.array([binomtest(k,n,.5).pvalue for k in range(n+1)]) for n in sizes}
  if len(labels)==2:
   threshold=a.mean(axis=0);ps=[table[int((y==c).sum())][(a[y==c]>threshold).sum(axis=0)] for c in labels]; scores=-np.minimum(*ps)
  else:
   ps=[]
   for c,d in itertools.combinations(labels,2):
    ac=a[y==c];ad=a[y==d]; threshold=np.median(np.concatenate([ac,ad]),axis=0)
    ps.append(np.minimum(table[len(ac)][(ac>threshold).sum(axis=0)],table[len(ad)][(ad>threshold).sum(axis=0)]))
   stat=-2*np.log(np.maximum(np.array(ps),1e-300)).sum(axis=0);scores=-chi2.sf(stat,2*len(ps))
 else:raise ValueError(method)
 scores=np.nan_to_num(scores,nan=-np.inf)
 # Original Python stable sorting retains source-column order at ties.
 order=np.argsort(-scores,kind='stable')
 return order,scores
