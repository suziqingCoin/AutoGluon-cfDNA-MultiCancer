from protocol import *
from fold import valid
import argparse,collections,math,re,unicodedata
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from scipy.stats import beta
NBOOT=2000
NOTE='Conditional on pooled fitted OOF predictions; excludes full model training/selection uncertainty and overlapping-fold dependence.'
def ci(vals):return list(map(float,np.quantile(vals,[.025,.975])))
def binary_ci(k,n):return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]
def boot(controls,cases):
 rng=np.random.RandomState(42);au=[];ap=[]
 for _ in range(NBOOT):
  c=rng.choice(controls,len(controls),replace=True);s=rng.choice(cases,len(cases),replace=True);y=np.r_[np.zeros(len(c)),np.ones(len(s))];p=np.r_[c,s];au.append(roc_auc_score(y,p));ap.append(average_precision_score(y,p))
 return ci(au),ci(ap)
def stage(x):
 if pd.isna(x):return 'Missing'
 v=unicodedata.normalize('NFKC',str(x)).upper().strip();v=re.sub(r'^STAGE\s*','',v);v=re.sub(r'[\s._-]+','',v)
 if v in ('','NA','N/A','NAN','NONE'):return 'Missing'
 for roman in ['III','II','I']:
  if re.fullmatch(roman+'[A-C]?',v):return roman
 return 'Unknown'
def clinical():
 names=['Group','Label','Source','SampleID','Age','Gender','Tumor Stage','Tumor type','Metastasis','Number of tumor','Tumor diameter']
 a=pd.read_excel(Path(os.environ.get('CFDNA_CLINICAL',SOURCE_ROOT.parent/'data/clinical.xlsx')),sheet_name='Table S1',header=None,skiprows=3,names=names,usecols=range(11),dtype=object)
 assert len(a)==2288 and a.SampleID.is_unique and a.SampleID.notna().all();a['stage']=a['Tumor Stage'].map(stage);return a

def threshold(c,spec):
 # >= threshold positive, smallest representable threshold meeting target specificity.
 return float(np.nextafter(np.sort(c)[math.ceil(spec*len(c))-1],np.inf))
def task_report(task):
 d,features=load(task);assert all(valid(task,i) for i in range(len(d))),f'{task}: incomplete or invalid; no final report'
 out=ROOT/'results'/task;out.mkdir(exist_ok=True)
 preds=[json.loads((ROOT/'folds'/task/f'{i:05d}'/'prediction.json').read_text()) for i in range(len(d))];a=pd.DataFrame(preds)
 assert a.SampleID.is_unique and set(a.SampleID)==set(d.SampleID) and len(a)==COUNTS[task] and not a.p_1.isna().any()
 assert np.array_equal(a.y_true,d.Label);a.to_csv(out/'oof_predictions.csv',index=False)
 c=a.loc[a.y_true.eq(0),'p_1'].to_numpy();s=a.loc[a.y_true.eq(1),'p_1'].to_numpy();au,ap=boot(c,s);y=a.y_true.to_numpy();p=a.p_1.to_numpy();pred=a.y_pred.to_numpy()
 overall=dict(task=task,n=len(a),n_controls=len(c),n_cancer=len(s),AUC=float(roc_auc_score(y,p)),AUC_CI_low=au[0],AUC_CI_high=au[1],AUPRC=float(average_precision_score(y,p)),AUPRC_CI_low=ap[0],AUPRC_CI_high=ap[1],Accuracy=float(accuracy_score(y,pred)),Precision=float(precision_score(y,pred,zero_division=0)),Recall=float(recall_score(y,pred)),F1=float(f1_score(y,pred)),Specificity=float((pred[y==0]==0).mean()),bootstrap_replicates=NBOOT,CI_note=NOTE,status='complete')
 pd.DataFrame([overall]).to_csv(out/'Table_S4.csv',index=False);pd.DataFrame([overall]).to_csv(out/'Figure_2C.csv',index=False)
 for col in ['method','model']:
  z=a[col].value_counts().rename_axis(col).rename('count').reset_index();z['frequency']=z['count']/len(a);z.to_csv(out/f'{col}_selection_frequency.csv',index=False)
 pd.DataFrame([dict(task=task,K_mean=a.k.mean(),K_median=a.k.median(),K_min=int(a.k.min()),K_max=int(a.k.max()))]).to_csv(out/'K_summary.csv',index=False);a[['fold','SampleID','k']].to_csv(out/'K_distribution.csv',index=False)
 selected=pd.concat([pd.read_csv(ROOT/'folds'/task/f'{i:05d}'/'selected_features.csv').assign(fold=i) for i in range(len(a))],ignore_index=True)
 freq=selected.feature.value_counts().reindex(features,fill_value=0).rename_axis('feature').rename('selected_folds').reset_index();freq['frequency']=freq.selected_folds/len(a);freq['family']=freq.feature.map(family);freq.to_csv(out/'feature_selection_frequency.csv',index=False)
 selected.groupby(['fold','family']).size().rename('n_selected').reset_index().to_csv(out/'feature_family_by_fold.csv',index=False)
 ff=freq.groupby('family').agg(total_selections=('selected_folds','sum'),features_ever_selected=('selected_folds',lambda v:int((v>0).sum()))).reset_index();ff['mean_features_per_fold']=ff.total_selections/len(a);ff.to_csv(out/'feature_family_summary.csv',index=False)
 cl=clinical();assert set(d.SampleID)<=set(cl.SampleID);merged=a.merge(cl[['SampleID','Label','Tumor Stage','stage']],on='SampleID',validate='one_to_one');assert np.array_equal(merged.Diagnosis,merged.Label);assert np.array_equal(merged.y_true,merged.Label.ne('Control').astype(int));merged.to_csv(out/'oof_with_clinical_stage.csv',index=False)
 merged.loc[merged.y_true.eq(1)].groupby('stage',dropna=False).size().rename('n').to_csv(out/'stage_counts_and_missingness.csv')
 stages=[];fixed=[]
 for group in ['I','II','I-II','III']:
  g=merged[merged.y_true.eq(1)&(merged.stage.isin(['I','II']) if group=='I-II' else merged.stage.eq(group))];ss=g.p_1.to_numpy();n=len(ss);small=n<10
  r=dict(task=task,stage=group,n_cases=n,n_controls=len(c),exploratory=small,AUC_reporting='suppressed n<10' if small else 'internal OOF',sensitivity_0_5=float((ss>=.5).mean()) if n else None)
  if n:
   low,high=binary_ci(int((ss>=.5).sum()),n);r.update(sensitivity_0_5_CI_low=low,sensitivity_0_5_CI_high=high)
  if n>=10:
   aci,_=boot(c,ss);r.update(AUC=float(roc_auc_score(np.r_[np.zeros(len(c)),np.ones(n)],np.r_[c,ss])),AUC_CI_low=aci[0],AUC_CI_high=aci[1])
  r.update(CI_note=NOTE);stages.append(r)
  for spec in [.95,.97]:
   th=threshold(c,spec);vals=[];rng=np.random.RandomState(42)
   if n:
    for _ in range(NBOOT):
     bc=rng.choice(c,len(c),replace=True);bs=rng.choice(ss,n,replace=True);vals.append(float((bs>=threshold(bc,spec)).mean()))
   lo,hi=ci(vals) if vals else [None,None]
   fixed.append(dict(task=task,stage=group,n_cases=n,n_controls=len(c),target_specificity=spec,threshold=th,achieved_specificity=float((c<th).mean()),sensitivity=float((ss>=th).mean()) if n else None,sensitivity_CI_low=lo,sensitivity_CI_high=hi,small_subgroup_exploratory=small,threshold_status='exploratory / retrospective pooled OOF-control threshold; NOT independent threshold validation',bootstrap_replicates=NBOOT,CI_note=NOTE))
 pd.DataFrame(stages).to_csv(out/'Table_S6.csv',index=False);pd.DataFrame(fixed).to_csv(out/'stage_fixed_specificity.csv',index=False)
 early=pd.DataFrame(fixed);early=early[early.stage.eq('I-II')];early.to_csv(out/'Table_S5.csv',index=False);early.to_csv(out/'Figure_2D.csv',index=False)
 dump(out/'integrity.json',dict(status='complete',n_predictions=len(a),expected=COUNTS[task],unique_patients=a.SampleID.nunique(),missing=0,duplicates=0,NaN=0,all_fold_artifacts_validated=True,raw_sha256=raw_sha(),signature=signature(),oof_sha256=sha(out/'oof_predictions.csv')))
 return overall

def combine():
 assert all((ROOT/'results'/t/'integrity.json').exists() for t in DIAG)
 for name in ['Table_S4','Table_S5','Table_S6','Figure_2C','Figure_2D']:
  frames=[pd.read_csv(ROOT/'results'/t/f'{name}.csv') for t in DIAG];pd.concat(frames,ignore_index=True).to_csv(ROOT/'results'/f'{name}.csv',index=False)
 with pd.ExcelWriter(ROOT/'results/manuscript_tables.xlsx') as writer:
  for name in ['Table_S4','Table_S5','Table_S6','Figure_2C','Figure_2D']:pd.read_csv(ROOT/'results'/f'{name}.csv').to_excel(writer,sheet_name=name,index=False)
 assert sum(json.loads((ROOT/'results'/t/'integrity.json').read_text())['n_predictions'] for t in DIAG)==8488
 dump(ROOT/'results/FINAL_COMPLETE.json',dict(expected=8488,successful_predictions=8488,status='complete',conditional_CI_note=NOTE))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--task',choices=list(DIAG));args=parser.parse_args()
 if not os.environ.get('CFDNA_NESTED_OUTPUT') or (ROOT.is_relative_to(SOURCE_ROOT.parent) and not ROOT.is_relative_to(SOURCE_ROOT.parent/'private')):
  raise SystemExit('Set CFDNA_NESTED_OUTPUT to a private/ or external output tree; committed results must remain unchanged')
 if args.task:task_report(args.task)
 else:
  for task in DIAG:task_report(task)
  combine()
