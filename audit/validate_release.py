"""Validate canonical public results without patient data or model training.

Writes a local check report to ignored audit/public_validation.json.
"""
from pathlib import Path
import ast,csv,hashlib,json,re,sys,tempfile
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
TASKS={'lung':'Lung_cancer','breast':'Breast_cancer','crc':'CRC_cancer','liver':'Liver_cancer','gastric':'Gastric_cancer','pan6':'pancancer'}
METHODS=['Fscore','ANOVA','ReliefF','Binomial']

def public_files():
    return sorted(p for p in R.rglob('*') if p.is_file() and p!=R/'audit/public_validation.json' and p.relative_to(R).parts[0] not in ['private','data','.venv','venv'] and not any(x in p.relative_to(R).parts for x in ['.git','__pycache__','public_reproduced','reproduced']))

def main():
    assert not (R/'tables').exists(),'Use canonical task/nested results instead of duplicate root tables/'
    totals=[];binary_candidates=[]
    for task,folder in TASKS.items():
        d=R/folder;c=json.loads((d/'code/config.json').read_text());assert c['strategy']=='none'
        v=pd.read_csv(d/'results/holdout/all_none_inner_validation_metrics.csv');assert len(v)==20 and v.strategy.eq('none').all()
        best=sorted(v.to_dict('records'),key=lambda r:tuple(-round(r[k],12) for k in ['AUC','AUPRC','Recall','Specificity'])+(r['k'],METHODS.index(r['method']),r['model']))[0]
        assert (best['method'],int(best['k']),best['model'])==(c['method'],c['k'],c['model'])
        f=pd.read_csv(d/'results/final_selected_features.csv');assert len(f)==c['k'] and f.feature.is_unique
        comparison=pd.read_csv(d/'results/tables/all_no_sampling_test_candidate_metrics.csv');assert len(comparison)==20 and comparison.selected_final_none.sum()==1
        sel=comparison[comparison.selected_final_none].iloc[0];assert (sel.method,sel.k,sel.model)==(c['method'],c['k'],c['model'])
        final=pd.read_csv(d/'results/tables/no_sampling_final_test_metrics_with_CI.csv');assert len(final)==1 and np.isclose(final.iloc[0].AUC,sel.AUC)
        for metric in ['AUC','AUPRC','Recall','Specificity']:
            row=final.iloc[0]
            assert np.isclose(row[metric],sel[metric]),(folder,metric)
            if metric+'_CI_low' in final:
                assert row[metric+'_CI_low']<=row[metric]<=row[metric+'_CI_high']
        conf=json.loads((d/'results/final_model_configuration.json').read_text())
        for k in ['hyperparameters_and_trained_parameters','inner_parameters','selection']:
            assert (d/'results'/conf[k]).is_file(),(folder,k)
        selected_dir=d/'results/holdout/none'/c['method']
        for name in ['actual_refit_fit_calls.json','exact_refit_input.json','refit_map.json']:
            assert (selected_dir/name).is_file(),(folder,name)
        split=json.loads((d/'results/split_reference.json').read_text())
        assert all(split[k]['n']>0 and len(split[k]['sha256'])==64 for k in ['inner_training','inner_validation','test'])
        assert (d/'results/holdout/ID_integrity.json').is_file()
        for method in METHODS:
            for name in ['features.json','feature_count_curve.csv','validation_metrics.csv']:
                assert (d/'results/holdout/none'/method/name).is_file()
        for item in json.loads((d/'figures/plot_manifest.json').read_text()):
            module=d/item['script'];assert module.is_file()
            tree=ast.parse(module.read_text())
            functions={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
            assert item['function'] in functions,(folder,item['function'])
            assert set(item.get('kwargs',{}))<={a.arg for a in functions[item['function']].args.args}
            assert item['supplementary_figure']=='S'+str(list(TASKS).index(task)+1)
            assert item['requires_private_data'] or not item['private_inputs_not_distributed']
            assert not re.search(r'cell\d+',module.name,re.I)
            for name in item['names']:assert (d/'figures'/f'{name}.pdf').is_file()
            for source in item['public_data']:assert (d/source).is_file()
            assert all('predictions' not in x for x in item['public_data'])
        if task!='pan6':
            binary_candidates.append(comparison)
            p=R/'nested_loocv/results'/task;x=pd.read_csv(p/'Table_S4.csv').iloc[0];totals.append(int(x.n))
            integrity=json.loads((p/'integrity.json').read_text());assert integrity['status']=='complete' and integrity['missing']==0 and integrity['duplicates']==0
            assert integrity['n_predictions']==x.n
            for fname in ['method_selection_frequency','model_selection_frequency','K_distribution_summary']:
                assert pd.read_csv(p/f'{fname}.csv')['count'].sum()==x.n
            hist=pd.read_csv(p/'K_distribution_summary.csv');k=pd.read_csv(p/'K_summary.csv').iloc[0]
            assert np.isclose(np.average(hist.k,weights=hist['count']),k.K_mean)
            assert np.isclose(hist.k.min(),k.K_min) and np.isclose(hist.k.max(),k.K_max)
    artifacts=pd.read_csv(R/'docs/paper_artifacts.csv').fillna('')
    for row in artifacts.to_dict('records'):
        if not row.get('private_function'):continue
        paths=row['entry_point'].split(';')
        for path in paths:assert (R/path).is_file(),path
        tree=ast.parse((R/paths[-1]).read_text())
        assert row['private_function'] in {n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
    assert sum(totals)==8488
    for k in [4,5,6]:
        merged=pd.concat([pd.read_csv(R/'nested_loocv/results'/task/f'Table_S{k}.csv') for task in list(TASKS)[:-1]],ignore_index=True)
        canonical=pd.read_csv(R/f'nested_loocv/results/Table_S{k}.csv')
        # Preserve source row order within each task while comparing the full combined report.
        pd.testing.assert_frame_equal(canonical.reset_index(drop=True),merged.reset_index(drop=True),check_dtype=False)
    s3=pd.concat(binary_candidates,ignore_index=True);assert len(s3)==100 and s3.selected_final_none.sum()==5
    per_class=pd.read_csv(R/'pancancer/results/tables/no_sampling_per_class_metrics_with_CI.csv')
    s7=per_class[per_class.dataset.eq('reused_holdout_test')];assert len(s7)==6
    cm=pd.read_csv(R/'pancancer/figures/data/Figure2F_confusion_counts.csv',index_col=0).to_numpy();assert cm.sum()==458
    assert np.array_equal(cm.sum(1),s7.support)
    assert np.allclose(cm.diagonal()/cm.sum(1),s7.Recall)
    sys.path.insert(0,str(R/'fig2'))
    import fig2
    from prepare_fig2_inputs import main as prepare
    with tempfile.TemporaryDirectory(prefix='cfdna-public-check-') as tmp:
        prepare(Path(tmp));checks=fig2.validate_data(fig2.read_inputs(Path(tmp)))
        for entry in json.loads((Path(tmp)/'source_manifest.json').read_text()):
            assert hashlib.sha256((R/entry['path']).read_bytes()).hexdigest()==entry['sha256']
    files=public_files()
    for p in files:
        if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
        assert p.stat().st_size<25*1024*1024,str(p.relative_to(R))
        assert p.suffix not in ['.pkl','.pickle','.pem','.key','.pth','.pt','.log','.joblib','.xlsx','.xls','.parquet'],str(p.relative_to(R))
        if p.suffix in ['.csv','.txt']:
            header=p.read_text().splitlines()[0] if p.stat().st_size else ''
            assert not re.search(r'(^|[,\t])(?:SampleID|y_true|y_pred|p_[0-5])([,\t]|$)',header),str(p.relative_to(R))
            if p.name.endswith('_coordinates.csv'):
                assert 'Threshold' not in header.split(','),str(p.relative_to(R))
        if p.suffix=='.ipynb':raise AssertionError('Historical notebook remains in public tree: '+str(p.relative_to(R)))
        if p.suffix in ['.py','.json','.md','.txt','.csv']:
            text=p.read_text();assert ('/'+'home'+'/'+'yjwei') not in text,str(p.relative_to(R))
            assert not re.search(r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}',text),str(p.relative_to(R))
        if p.suffix=='.md':
            for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
                if '://' in target or target.startswith('#'):continue
                assert (p.parent/target.split('#')[0]).exists(),(str(p.relative_to(R)),target)
    manifest=R/'audit/release_manifest.csv'
    assert manifest.is_file(),'Missing canonical file manifest'
    with manifest.open() as f:
        entries=list(csv.DictReader(f))
    assert len(entries)==len({e['path'] for e in entries}),'Duplicate manifest paths'
    for entry in entries:
        p=R/entry['path'];assert p.is_file(),entry['path']
        assert p.stat().st_size==int(entry['bytes']),entry['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'],entry['path']
    expected={p.relative_to(R).as_posix() for p in files if p!=manifest and p!=R/'audit/public_validation.json'}
    assert expected=={e['path'] for e in entries},'Public manifest coverage mismatch'
    result=dict(status='PASS',tasks=6,validation_locked_choices=6,test_candidates=120,successful_nested_outer_folds=sum(totals),table_sources='Canonical task and nested results; no tables/ duplicates',figure2_data=checks,patient_tables_present=False,python_syntax='PASS',markdown_links='PASS',manifest_files=len(entries),public_files_checked=len(files),scope='Public aggregates, paths, hashes and static hygiene only. Private-source checks are prior evidence, not a new patient-level replication; Git history requires a separate review.')
    (R/'audit/public_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: six validation-selected recipes, 120 candidates, 8488 nested folds, canonical table sources, Figure 2, paths, hashes, syntax and public-file hygiene')
if __name__=='__main__':main()
