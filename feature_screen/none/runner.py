"""Run formal none without importing historical sampling notebooks."""
from pathlib import Path
import argparse,json,os,hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def main(config):
    config=Path(config);c=json.loads(config.read_text());repo=config.parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=os.environ.get('CFDNA_RAW'))
    parser.add_argument('--output-dir',type=Path,default=repo/'private/none'/c['task'])
    parser.add_argument('--check-config',action='store_true',help='Import dependencies and validate configuration; no fitting')
    parser.add_argument('--dry-run',action='store_true',help='Validate raw input, original split hashes and labels; no fitting')
    parser.add_argument('--training-cv',action='store_true',help='Also run original training-only three-fold comparison')
    args=parser.parse_args()
    import none_experiment as m
    assert c['strategy']=='none' and m.STRATEGIES==['none']
    assert c['method'] in m.METHODS and c['k']>0
    if args.check_config:
        print(c['task'],'none-only configuration and imports PASS; no training');return
    if args.data is None or not args.data.is_file():parser.error('Provide an authorized cohort CSV via --data or CFDNA_RAW; see DATA_ACCESS.md')
    data=pd.read_csv(args.data);features=data.columns[4:].tolist()
    assert list(data.columns[:4])==['SampleID','Cohort','Label','Diagnosis'], 'Preserve the original first four metadata columns and molecular column order'
    assert data.SampleID.notna().all() and data.SampleID.is_unique
    assert len(features)==8630 and np.isfinite(data[features].to_numpy()).all()
    if c['task']=='pan6':
        names=m.NAMES;data=data.copy();data['__target__']=data.Diagnosis.map(dict(zip(names,range(6))))
    else:
        data=data[data.Diagnosis.isin(['Control',m.DIAG[c['task']]])].copy();names=['Control',m.DIAG[c['task']]];data['__target__']=data.Label
        assert np.array_equal(data.__target__.to_numpy(),data.Diagnosis.ne('Control').astype(int).to_numpy())
    data=data.reset_index(drop=True);assert data.__target__.notna().all()
    tr,te=train_test_split(np.arange(len(data)),test_size=.2,stratify=data.__target__,random_state=42)
    inn,va=train_test_split(tr,test_size=.2,stratify=data.iloc[tr].__target__,random_state=43 if c['task']=='pan6' else 42)
    ref=json.loads((config.parent.parent/'results/split_reference.json').read_text())
    for label,idx in [('inner_training',inn),('inner_validation',va),('test',te)]:
        ids=sorted(data.iloc[idx].SampleID.astype(str));assert len(ids)==ref[label]['n']
        assert hashlib.sha256('\n'.join(ids).encode()).hexdigest()==ref[label]['sha256'],label
    print(c['task'],'raw schema, labels, patient isolation and all original split hashes PASS')
    if args.dry_run:return
    # Fail closed on an occupied output: never overwrite published results or resume foreign artifacts.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):parser.error('Use a new empty output directory; preserve existing analysis artifacts')
    m.ROOT=args.output_dir.resolve();m.ROOT.mkdir(parents=True,exist_ok=True);m.RAW=args.data.resolve()
    (m.ROOT/'split_reference.json').write_text(json.dumps(ref,indent=2))
    import shutil
    shutil.copy2(repo/'feature_screen/none/PROTOCOL.md',m.ROOT/'protocol_locked_before_training.md')
    if args.training_cv:m.run_task(c['task'])
    else:m.seed();m.context_run(data,features,names,c['task'],'holdout',tr,te,True)
if __name__=='__main__':raise SystemExit('Use <task>/code/run_none.py')
