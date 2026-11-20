from protocol import *
import sys,time,traceback,shutil,fcntl
from sklearn.model_selection import train_test_split
from feature_rankings import rank

def valid(task,i):
 try:
  p=ROOT/'folds'/task/f'{i:05d}';m=json.loads((p/'SUCCESS.json').read_text());a=json.loads((p/'prediction.json').read_text())
  if m['signature']!=signature() or m['raw_sha256']!=raw_sha():return False
  if not all(sha(p/f)==h for f,h in m['artifacts'].items()):return False
  d,features=load(task);split=pd.read_csv(p/'split.csv');ff=pd.read_csv(p/'selected_features.csv')
  assert a['SampleID']==d.iloc[i].SampleID and a['y_true']==int(d.iloc[i].Label) and a['fold']==i and a['status']=='success'
  assert np.isfinite(a['p_1']) and 0<=a['p_1']<=1 and abs(a['p_0']+a['p_1']-1)<1e-6
  assert len(ff)==a['k'] and ff.feature.is_unique and set(ff.feature)<=set(features)
  assert len(split)==len(d) and split.SampleID.is_unique and set(split.SampleID)==set(d.SampleID)
  assert split.loc[split.part.eq('outer_test'),'SampleID'].tolist()==[a['SampleID']]
  assert len(split[split.part.eq('outer_test')])==1
  assert m['n_actual_outer_fit']==len(d)-1 and m['n_candidates']==20
  return True
 except Exception:return False

def run(task,i):
 p=ROOT/'folds'/task/f'{i:05d}';p.mkdir(parents=True,exist_ok=True)
 with (p/'fold.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if valid(task,i):print('SKIP',task,i,flush=True);return
  start=time.time();scratch=ROOT/'scratch'/f'{task}_{i:05d}';shutil.rmtree(scratch,ignore_errors=True);scratch.mkdir()
  try:
   from autogluon.tabular import TabularPredictor
   from autogluon.features.generators import AutoMLPipelineFeatureGenerator
   from threadpoolctl import threadpool_limits,threadpool_info
   with threadpool_limits(limits=1):
    seed();d,features=load(task);outer=np.delete(np.arange(len(d)),i);inn,va=train_test_split(outer,test_size=.2,stratify=d.iloc[outer].Label,random_state=42)
    I=d.iloc[inn];V=d.iloc[va];T=d.iloc[outer];E=d.iloc[[i]]
    for a,b in [(I,V),(I,E),(V,E),(T,E)]:assert not set(a.SampleID)&set(b.SampleID)
    split=d[['SampleID','Label','Diagnosis']].copy();split['part']='inner_train';split.loc[va,'part']='inner_validation';split.loc[i,'part']='outer_test';split.to_csv(p/'split.csv',index=False)
    rows=[];predictors={}
    for method,cap in zip(METHODS,CAPS[task]):
     print(task,i,method,'ranking/K/model search',flush=True);mdir=p/method;mdir.mkdir(exist_ok=True)
     order,scores=rank(I[features],I.Label.to_numpy(),method)
     pd.DataFrame(dict(feature=np.array(features)[order],score=scores[order])).to_csv(mdir/'inner_ranking.csv',index=False)
     k=optimize(I[features].to_numpy(),I.Label.to_numpy(),V[features].to_numpy(),V.Label.to_numpy(),order,cap,mdir/'K_curve.csv');cols=np.array(features)[order[:k]].tolist()
     seed();train=I[cols].reset_index(drop=True);train['target']=I.Label.to_numpy();val=V[cols].reset_index(drop=True);val['target']=V.Label.to_numpy()
     # Fit unsupervised feature generator only on inner training, not tuning_data.
     gen=AutoMLPipelineFeatureGenerator();gen.fit_transform(train[cols])
     ag=TabularPredictor(label='target',problem_type='binary',eval_metric='roc_auc',sample_weight=None,path=str(scratch/method),verbosity=1)
     ag.fit(train_data=train,tuning_data=val,feature_generator=gen,hyperparameters={'XGB':{},'RF':{'criterion':'gini'},'GBM':{},'FASTAI':{},'ENS_WEIGHTED':{}},num_bag_folds=0,num_stack_levels=0,dynamic_stacking=False,num_cpus=1,num_gpus=0,ag_args_fit={'num_cpus':1,'num_gpus':0})
     assert set(ag.model_names())==MODELS,ag.model_names();assert ag.class_labels==[0,1]
     for model in ag.model_names():
      pred=ag.predict_proba(val[cols],model=model,as_multiclass=True)[1].to_numpy();r=score(V.Label.to_numpy(),pred);r.update(method=method,k=k,model=model);rows.append(r)
     dump(mdir/'inner_model_parameters.json',ag.info());predictors[method]=ag
    assert len(rows)==20;best=choose(rows);dump(p/'LOCKED_SELECTION.json',dict(best=best,n_candidates=20,outer_prediction_seen=False,selection_uses='inner_validation_only'))
    pd.DataFrame(rows).to_csv(p/'candidates.csv',index=False)
    # Recompute supervision on ALL and ONLY outer training after locking method/K/model.
    method=best['method'];order,scores=rank(T[features],T.Label.to_numpy(),method);cols=np.array(features)[order[:best['k']]].tolist()
    pd.DataFrame(dict(feature=cols,family=[family(f) for f in cols],rank=np.arange(1,len(cols)+1),score=scores[order[:best['k']]])).to_csv(p/'selected_features.csv',index=False)
    pd.DataFrame(dict(feature=np.array(features)[order],score=scores[order])).to_csv(p/'outer_training_ranking.csv',index=False)
    # New outer preprocessing is fit only on outer training; base templates inherit
    # inner-selected iteration/epoch counts, never inner-selected feature identities.
    ag=predictors[method];gen=AutoMLPipelineFeatureGenerator();X=gen.fit_transform(T[cols].reset_index(drop=True));XT=gen.transform(E[cols].reset_index(drop=True));y=T.Label.reset_index(drop=True)
    if best['model'].startswith('WeightedEnsemble'):
     weights=ag._trainer.load_model(best['model'])._get_model_weights()
    else:weights={best['model']:1.}
    assert abs(sum(weights.values())-1)<1e-6
    prob=0.;fitproof=[]
    for name,weight in weights.items():
     seed();parent=ag._trainer.load_model(name);model=parent.convert_to_refit_full_template();model.set_contexts(str(scratch/'outer'/name))
     assert not model.is_fit();model.fit(X=X.copy(),y=y.copy(),X_val=None,y_val=None,num_cpus=1,num_gpus=0)
     assert model.is_fit()
     q=np.asarray(model.predict_proba(XT));assert q.shape==(1,)
     prob+=float(weight)*float(q[0]);fitproof.append(dict(model=name,weight=weight,n_actual_fit=len(X),n_unique_IDs=T.SampleID.nunique(),params=model.get_info(),sample_weight=None))
    assert np.isfinite(prob) and 0<=prob<=1
    a=dict(task=task,fold=i,SampleID=E.iloc[0].SampleID,Diagnosis=E.iloc[0].Diagnosis,y_true=int(E.iloc[0].Label),p_0=1-prob,p_1=prob,y_pred=int(prob>.5),method=method,k=best['k'],model=best['model'],status='success',elapsed_seconds=time.time()-start)
    dump(p/'prediction.json',a);dump(p/'fit_proof.json',dict(outer_fit=fitproof,inner_training_n=len(I),inner_validation_n=len(V),outer_training_n=len(T),outer_test_n=1,overlaps=0,preprocessing='separate fit on inner/outer training only',threads=threadpool_info(),ensemble_weights_fit='inner validation only; locked weights applied to refitted outer bases'))
    artifacts=['prediction.json','fit_proof.json','split.csv','selected_features.csv','outer_training_ranking.csv','LOCKED_SELECTION.json','candidates.csv']+[f'{m}/{f}' for m in METHODS for f in ['K_curve.csv','inner_ranking.csv','inner_model_parameters.json']]
    dump(p/'SUCCESS.json',dict(signature=signature(),raw_sha256=raw_sha(),n_candidates=20,n_actual_outer_fit=len(T),artifacts={f:sha(p/f) for f in artifacts},elapsed_seconds=time.time()-start))
    assert valid(task,i);print('SUCCESS',task,i,a,flush=True)
  except Exception as e:
   (p/'SUCCESS.json').unlink(missing_ok=True);dump(p/f'failure_{time.time_ns()}.json',dict(error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc()));raise
  finally:shutil.rmtree(scratch,ignore_errors=True)
if __name__=='__main__':run(sys.argv[1],int(sys.argv[2]))
