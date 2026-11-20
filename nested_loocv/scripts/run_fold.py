"""Initialize a fresh private output tree and invoke the authentic single-fold worker."""
from pathlib import Path
import argparse,sys
p=argparse.ArgumentParser(description=__doc__);p.add_argument('task',choices=['lung','breast','crc','liver','gastric']);p.add_argument('fold',type=int)
a=p.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from protocol import ROOT,COUNTS,SOURCE_ROOT,os,json,signature
from fold import run
if not os.environ.get('CFDNA_NESTED_OUTPUT') or (ROOT.is_relative_to(SOURCE_ROOT.parent) and not ROOT.is_relative_to(SOURCE_ROOT.parent/'private')):
 raise SystemExit('Set CFDNA_NESTED_OUTPUT to a separate private output tree')
previous=ROOT/'audit/runtime_config.json'
if previous.exists() and json.loads(previous.read_text()).get('signature')!=signature():
 raise SystemExit('Existing output has a different source signature; use a fresh output directory')
assert 0<=a.fold<COUNTS[a.task]
for sub in ['folds','logs','scratch','audit','results']:(ROOT/sub).mkdir(parents=True,exist_ok=True)
run(a.task,a.fold)
