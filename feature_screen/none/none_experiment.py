"""Formal none strategy; patient holdout and training-only internal selection.
Portability and verification scope: docs/REPRODUCIBILITY.md and audit/algorithm_equivalence.json.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,random,traceback,argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PREV=ROOT
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import roc_auc_score,average_precision_score,recall_score,precision_score,accuracy_score,f1_score,confusion_matrix
from none_protocol import METHODS,CAPS,PARAMS,balance,bootstrap_auc,dump
from feature_rankings import rank
from autogluon.tabular import TabularPredictor
import torch
TASKS=['lung','breast','crc','liver','gastric','pan6'];STRATEGIES=['none']
DIAG={'lung':'Lung cancer','breast':'Breast cancer','crc':'CRC cancer','liver':'Liver cancer','gastric':'Gastric cancer'}
NAMES=['Control','Breast cancer','CRC cancer','Gastric cancer','Liver cancer','Lung cancer']
RAW=Path(os.environ.get('CFDNA_RAW',ROOT/'data/cohort.csv'))
def seed():
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(1)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass

def prepare(df,task,strategy):
 assert strategy=='none', 'Public formal entry supports none only'
 b=df.copy().reset_index(drop=True);b['__weight__']=1.
 return b

def score(y,p):
 pred=p.argmax(axis=1);healthy=y==0
 out=dict(AUC=float(roc_auc_score(y,p[:,1])) if p.shape[1]==2 else float(roc_auc_score(y,p,multi_class='ovr',average='weighted')),AUPRC=float(average_precision_score(y,p[:,1])) if p.shape[1]==2 else float(average_precision_score(np.eye(p.shape[1])[y],p,average='weighted')),Recall=float(recall_score(y,pred,average='binary' if p.shape[1]==2 else 'macro',zero_division=0)),Specificity=float((pred[healthy]==0).mean()),Accuracy=float(accuracy_score(y,pred)),Precision=float(precision_score(y,pred,average='binary' if p.shape[1]==2 else 'macro',zero_division=0)),F1=float(f1_score(y,pred,average='binary' if p.shape[1]==2 else 'macro',zero_division=0)))
 if p.shape[1]>2:
  out['macro_AUC']=float(roc_auc_score(y,p,multi_class='ovr',average='macro'));out['macro_AUPRC']=float(average_precision_score(np.eye(p.shape[1])[y],p,average='macro'));cancer=y>0;out['cancer_detection_Recall']=float((pred[cancer]>0).mean());out['cancer_exact_type_Recall']=float((pred[cancer]==y[cancer]).mean())
 return out

def choose(rows):
 # Rounding only treats machine precision differences as ties, as prospectively locked.
 return sorted(rows,key=lambda r:tuple(-round(r[k],12) for k in ['AUC','AUPRC','Recall','Specificity'])+(r['k'],METHODS.index(r['method']),r['model'],{'none':0,'class_weight':1,'resample':2}[r['strategy']]))[0]

def probs(predictor,X,model,nclass):return predictor.predict_proba(X,model=model,as_multiclass=True).loc[:,list(range(nclass))].to_numpy()

def optimization(X,y,w,Xv,yv,order,cap,path,nclass):
 params=dict(PARAMS,objective='binary:logistic' if nclass==2 else 'multi:softprob',eval_metric='auc' if nclass==2 else 'mlogloss')
 if nclass>2:params['num_class']=nclass
 rows=[];best=-np.inf;bestk=1;over=decline=0;prev=0
 for k in range(1,min(cap,len(order))+1):
  cols=order[:k];dt=xgb.DMatrix(X[:,cols],label=y,weight=w);dv=xgb.DMatrix(Xv[:,cols],label=yv)
  model=xgb.train(params,dt,num_boost_round=50,evals=[(dt,'train'),(dv,'eval')],early_stopping_rounds=10,verbose_eval=False)
  pt=model.predict(dt);pv=model.predict(dv)
  at=roc_auc_score(y,pt) if nclass==2 else roc_auc_score(y,pt,multi_class='ovr',average='weighted');av=roc_auc_score(yv,pv) if nclass==2 else roc_auc_score(yv,pv,multi_class='ovr',average='weighted')
  rows.append(dict(k=k,train_auc=float(at),validation_auc=float(av),best_iteration=model.best_iteration))
  if av>best:best=float(av);bestk=k
  over=over+1 if at-av>.1 else 0;decline=decline+1 if av<prev else 0;prev=av
  if over>=10 or decline>=5:break
 pd.DataFrame(rows).to_csv(path,index=False);return bestk,best

def trace_sampling(df,b,out,stage):
 b[['SampleID','Diagnosis','__target__','__weight__']].to_csv(out/f'{stage}_sampled_IDs.csv',index=False)
 rows=[]
 for name,g in df.groupby('Diagnosis',sort=False):
  a=b[b.Diagnosis.eq(name)];rows.append(dict(stage=stage,Diagnosis=name,before_records=len(g),before_unique_IDs=g.SampleID.nunique(),after_records=len(a),after_unique_IDs=a.SampleID.nunique(),effective_multiplier=len(a)/len(g),weight_min=float(a.__weight__.min()),weight_max=float(a.__weight__.max())))
 pd.DataFrame(rows).to_csv(out/f'{stage}_sampling_counts.csv',index=False)

def exact_refit(predictor,final,selected,models,out):
 """Use version-locked advanced arguments; no default cached validation append."""
 # Resume safety: a process interrupted after AG saved _FULL but before our map
 # checkpoint must actually refit again so the monitored input proof is complete.
 existing=[n for n in predictor.model_names() if n.endswith('_FULL')]
 if existing:
  import shutil
  assert str(Path(predictor.path).resolve()).startswith(str((ROOT/'results').resolve()))
  snapshot=out/('refit_recovery_snapshot_'+str(time.time_ns()))
  shutil.copytree(predictor.path,snapshot)
  predictor.delete_models(models_to_delete=existing,dry_run=False)
  dump(out/'refit_recovery_note.json',dict(reason='interrupted before verified refit_map checkpoint',archived_predictor=str(snapshot),removed_only_new_analysis_FULL_models=existing,scientific_parameters_changed=False))
 X=predictor.transform_features(final[selected]).reset_index(drop=True)
 if predictor.sample_weight is not None:X[predictor.sample_weight]=final['__weight__'].to_numpy()
 y=predictor.transform_labels(final['__target__']).reset_index(drop=True)
 from autogluon.core.models.abstract.abstract_model import AbstractModel
 original_fit=AbstractModel.fit;calls=[]
 def monitored_fit(obj,*args,**kw):
  xx=kw.get('X',args[0] if args else None);yy=kw.get('y',args[1] if len(args)>1 else None)
  if obj.name.endswith('_FULL'):
   assert len(xx)==len(final) and np.array_equal(np.asarray(yy),np.asarray(y)), 'Refit actual input differs from intended strategy'
   sw=kw.get('sample_weight');calls.append(dict(model=obj.name,n_actual_fit_rows=len(xx),y_counts=pd.Series(yy).value_counts().to_dict(),weight_min=None if sw is None else float(np.min(sw)),weight_max=None if sw is None else float(np.max(sw))))
   if predictor.sample_weight is not None:assert sw is not None and np.allclose(sw,final.__weight__.to_numpy(),rtol=1e-6)
  return original_fit(obj,*args,**kw)
 AbstractModel.fit=monitored_fit
 try:maps=predictor.refit_full(model=models,set_best_to_refit_full=False,X=X,y=y,X_val=X.iloc[:0].copy(),y_val=y.iloc[:0].copy())
 finally:AbstractModel.fit=original_fit
 dump(out/'actual_refit_fit_calls.json',calls)
 info=predictor.info();dump(out/'fitted_model_parameters.json',info)
 dump(out/'exact_refit_input.json',dict(n_records=len(X),n_unique_IDs=final.SampleID.nunique(),label_counts=final.__target__.value_counts().to_dict(),X_val_rows=0,advanced_API='AutoGluon 1.1.1 refit_full(X=transformed_target_train,y=encoded_labels,X_val=empty,y_val=empty)',feature_generator_fit='inner_training (AutoGluon fit may use unlabeled inner validation for feature generation)',sample_weight_column=predictor.sample_weight))
 for parent in models:
  assert parent in maps
  if not parent.startswith('WeightedEnsemble'):
   child=predictor._trainer.load_model(maps[parent]);assert child.get_info()['is_fit']
   # A failed refit would silently clone parent in AG; assert it has a genuine fit runtime and reset input dimensions.
   assert child.fit_time is not None and any(c['model']==maps[parent] for c in calls), 'AG silently cloned a base model instead of refitting'
  if parent=='NeuralNetFastAI' and predictor.sample_weight is not None:
   child=predictor._trainer.load_model(maps[parent]);expected=weights(final['__target__']);actual=child.params_trained['training_class_weights'];encoded=y.to_numpy()
   assert np.allclose(np.array(actual)[encoded],expected,rtol=1e-6)
 return maps

def export_pred(path,df,p,task,context,strategy,method,model):
 a=df[['SampleID','Diagnosis','__target__']].rename(columns={'__target__':'y_true'}).copy().reset_index(drop=True);a['y_pred']=p.argmax(axis=1)
 for c in range(p.shape[1]):a[f'p_{c}']=p[:,c]
 a['task']=task;a['context']=context;a['strategy']=strategy;a['method']=method;a['model']=model;a.to_csv(path,index=False)


def context_run(data,features,names,task,context,trainidx,evalidx,is_holdout):
 out=ROOT/'results'/task/context;out.mkdir(parents=True,exist_ok=True)
 if (out/'COMPLETE.json').exists():print(task,context,'SKIP COMPLETE',flush=True);return
 inn,va=train_test_split(trainidx,test_size=.2,stratify=data.iloc[trainidx].__target__,random_state=43 if task=='pan6' else 42)
 I=data.iloc[inn].copy();V=data.iloc[va].copy();T=data.iloc[trainidx].copy();Eval=data.iloc[evalidx].copy();yI=I.__target__.to_numpy();yV=V.__target__.to_numpy()
 for a,b in [(I,V),(I,Eval),(V,Eval)]:assert not set(a.SampleID)&set(b.SampleID)
 split=data.iloc[np.r_[inn,va,evalidx]][['SampleID','Diagnosis','__target__']].copy();split['part']=['inner_training']*len(inn)+['inner_validation']*len(va)+['evaluation']*len(evalidx);split.to_csv(out/'patient_split.csv',index=False)
 if is_holdout:
  # Verify the actual original split against public hash commitments, without exposing IDs.
  import hashlib
  reference=ROOT/'split_reference.json'
  if reference.exists():
   checks=json.loads(reference.read_text())
   for label,arr in [('inner_training',inn),('inner_validation',va),('test',evalidx)]:
    digest=hashlib.sha256('\n'.join(sorted(data.iloc[arr].SampleID.astype(str))).encode()).hexdigest()
    assert digest==checks[label]['sha256'],label
 dump(out/'ID_integrity.json',dict(task=task,context=context,inner_train_validation_overlap=0,inner_train_eval_overlap=0,validation_eval_overlap=0,train_eval_overlap=len(set(T.SampleID)&set(Eval.SampleID)),n_original_train=len(T),n_original_evaluation=len(Eval),n_inner_train=len(I),n_inner_validation=len(V),matches_previous_split=is_holdout))
 # Shared rankings are fitted once on the raw inner-training patients, never outer-CV validation.
 rankings={}
 for method in METHODS:
  rdir=out/'rankings';rdir.mkdir(exist_ok=True);p=rdir/f'{method}.csv'
  if p.exists():r=pd.read_csv(p);lookup={f:i for i,f in enumerate(features)};order=np.array([lookup[f] for f in r.feature])
  else:
   print(task,context,method,'rank',flush=True);order,scores=rank(I[features],yI,method);pd.DataFrame({'feature':np.array(features)[order],'score':scores[order]}).to_csv(p,index=False)
  rankings[method]=order
 allrows=[]
 for strategy in STRATEGIES:
  sdir=out/strategy;sdir.mkdir(exist_ok=True);B=prepare(I,task,strategy);final=prepare(T,task,strategy)
  assert set(B.SampleID)<=set(I.SampleID) and not set(B.SampleID)&set(V.SampleID) and not set(final.SampleID)&set(Eval.SampleID)
  trace_sampling(I,B,sdir,'inner_training');trace_sampling(T,final,sdir,'final_training')
  X=B[features].to_numpy();y=B.__target__.to_numpy();w=B.__weight__.to_numpy() if strategy=='class_weight' else None
  for method,cap in zip(METHODS,CAPS[task]):
   mdir=sdir/method;mdir.mkdir(exist_ok=True)
   if (mdir/'validation_metrics.csv').exists():allrows+=pd.read_csv(mdir/'validation_metrics.csv').to_dict('records');continue
   print(task,context,strategy,method,'optimize / fit',flush=True)
   order=rankings[method]
   if (mdir/'features.json').exists():selection=json.loads((mdir/'features.json').read_text());selected=selection['features'];k=selection['k']
   else:
    k,val=optimization(X,y,w,V[features].to_numpy(),yV,order,cap,mdir/'feature_count_curve.csv',len(names));selected=np.array(features)[order[:k]].tolist();selection=dict(k=k,features=selected,ranking_fit_n=len(I),validation_xgb_auc=val,cap=cap);dump(mdir/'features.json',selection)
   seed();train=B[selected+['__target__']].copy();valid=V[selected+['__target__']].copy();weightcol=None
   hp={'XGB':{},'RF':{'criterion':'gini'},'GBM':{},'FASTAI':{},'ENS_WEIGHTED':{}}
   if (mdir/'autogluon').exists():
    (mdir/'autogluon').rename(mdir/('interrupted_autogluon_'+str(time.time_ns())))
   predictor=TabularPredictor(label='__target__',problem_type='binary' if len(names)==2 else 'multiclass',eval_metric='roc_auc' if len(names)==2 else 'log_loss',sample_weight=weightcol,weight_evaluation=False,path=str(mdir/'autogluon'),verbosity=2)
   predictor.fit(train_data=train,tuning_data=valid,hyperparameters=hp,num_bag_folds=0,num_stack_levels=0,dynamic_stacking=False,num_cpus=2,num_gpus=0,ag_args_fit={'num_cpus':2,'num_gpus':0})
   models=predictor.model_names();assert len(models)==5 and set(models)=={'LightGBM','RandomForest','XGBoost','NeuralNetFastAI','WeightedEnsemble_L2'},models
   vals=[]
   for model in models:
    p=probs(predictor,valid,model,len(names));m=score(yV,p);m.update(task=task,context=context,strategy=strategy,method=method,model=model,k=k);vals.append(m)
    export_pred(mdir/f'validation_predictions_{model}.csv',V,p,task,context,strategy,method,model)
   dump(mdir/'fitted_inner_model_parameters.json',predictor.info());pd.DataFrame(vals).to_csv(mdir/'validation_metrics.csv',index=False);allrows+=vals
   torch.set_num_threads(1)
 # Selection is prospectively locked before any evaluation probabilities.
 bests={s:choose([r for r in allrows if r['strategy']==s]) for s in STRATEGIES};best=choose(list(bests.values()))
 dump(out/'LOCKED_SELECTION.json',dict(protocol_sha256=hashlib.sha256((ROOT/'protocol_locked_before_training.md').read_bytes()).hexdigest(),overall=best,per_strategy=bests,n_candidates=len(allrows),evaluation_seen=False))
 assert len(allrows)==len(STRATEGIES)*len(METHODS)*5
 pd.DataFrame(allrows).to_csv(out/'all_inner_validation_metrics.csv',index=False)
 metrics=[]
 for strategy in STRATEGIES:
  final=prepare(T,task,strategy);sdir=out/strategy
  evaluate_rows=[r for r in allrows if r['strategy']==strategy] if is_holdout else [bests[strategy]]
  for method in METHODS:
   cands=[r for r in evaluate_rows if r['method']==method]
   if not cands:continue
   mdir=sdir/method;selection=json.loads((mdir/'features.json').read_text());selected=selection['features'];predictor=TabularPredictor.load(str(mdir/'autogluon'),verbosity=0);seed()
   mapfile=mdir/'refit_map.json'
   if mapfile.exists():maps=json.loads(mapfile.read_text())
   else:
    models=[r['model'] for r in cands];maps=exact_refit(predictor,final,selected,models,mdir);dump(mapfile,maps)
   for cand in cands:
    p=probs(predictor,Eval[selected],maps[cand['model']],len(names));assert np.isfinite(p).all() and np.allclose(p.sum(axis=1),1,atol=1e-5)
    name=cand['model'];m=score(Eval.__target__.to_numpy(),p);m.update(task=task,context=context,strategy=strategy,method=method,model=name,k=cand['k'],selected_for_strategy=name==bests[strategy]['model'] and method==bests[strategy]['method'],selected_overall=name==best['model'] and method==best['method'] and strategy==best['strategy']);metrics.append(m)
    export_pred(mdir/f'evaluation_predictions_{name}.csv',Eval,p,task,context,strategy,method,name)
    if m['selected_for_strategy']:export_pred(sdir/'selected_evaluation_predictions.csv',Eval,p,task,context,strategy,method,name)
    cm=pd.DataFrame(confusion_matrix(Eval.__target__,p.argmax(axis=1),labels=list(range(len(names)))),index=names,columns=names);cm.to_csv(mdir/f'evaluation_confusion_{name}.csv')
    if m['selected_overall']:export_pred(out/'selected_policy_predictions.csv',Eval,p,task,context,strategy,method,name)
 pd.DataFrame(metrics).to_csv(out/'all_evaluation_metrics.csv',index=False)
 dump(out/'COMPLETE.json',dict(task=task,context=context,n_candidates=len(allrows),n_evaluated=len(metrics),status='complete',chosen_strategy=best['strategy']))
 print(task,context,'COMPLETE strategy',best['strategy'],flush=True)

def run_task(task):
 if (ROOT/'results'/task/'TASK_COMPLETE.json').exists():return
 seed();raw=pd.read_csv(RAW);features=raw.columns[4:].tolist();assert raw.SampleID.is_unique
 if task=='pan6':data=raw.copy();names=NAMES;data['__target__']=data.Diagnosis.map(dict(zip(names,range(6))))
 else:data=raw[raw.Diagnosis.isin(['Control',DIAG[task]])].copy();names=['Control',DIAG[task]];data['__target__']=data.Label
 data=data.reset_index(drop=True);tr,te=train_test_split(np.arange(len(data)),test_size=.2,stratify=data.__target__,random_state=42)
 context_run(data,features,names,task,'holdout',tr,te,True)
 kfold=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
 for fold,(a,b) in enumerate(kfold.split(tr,data.iloc[tr].__target__)):
  context_run(data,features,names,task,f'nested_fold_{fold}',tr[a],tr[b],False)
 out=ROOT/'results'/task;dump(out/'TASK_COMPLETE.json',dict(task=task,n_contexts=4,status='complete'))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--tasks',nargs='+',default=TASKS);args=parser.parse_args()
 for task in args.tasks:
  try:
   import fcntl
   lockpath=ROOT/'results'/task;lockpath.mkdir(parents=True,exist_ok=True)
   with (lockpath/'worker.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX);run_task(task)
  except Exception:
   p=ROOT/'results'/task;p.mkdir(parents=True,exist_ok=True);(p/'failure.log').write_text(traceback.format_exc());raise
