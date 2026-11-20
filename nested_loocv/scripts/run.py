import os,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from protocol import *
R=ROOT
if not os.environ.get('CFDNA_NESTED_OUTPUT') or (R.is_relative_to(SOURCE_ROOT.parent) and not R.is_relative_to(SOURCE_ROOT.parent/'private')):
 raise SystemExit('Set CFDNA_NESTED_OUTPUT to a separate private output tree; committed results must remain unchanged')
previous=R/'audit/runtime_config.json'
if previous.exists() and json.loads(previous.read_text()).get('signature')!=signature():
 raise SystemExit('Existing output belongs to a different source signature; choose a fresh output directory')
for sub in ['folds','logs','scratch','audit','results']:(R/sub).mkdir(parents=True,exist_ok=True)
from fold import valid
import subprocess,time,collections,fcntl,signal
import psutil

def capacity():
 cpu=len(os.sched_getaffinity(0));mem=psutil.virtual_memory().available/2**30;load=os.getloadavg()[0]
 return max(1,min(12,int(max(1,cpu-load-8)),int(max(1,mem-48)//10)))
def main():
 with (R/'runner.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  (R/'main.pid').write_text(str(os.getpid()));sig=signature();done={t:set() for t in DIAG};pending=collections.deque();attempts=collections.Counter();active={};failed={};reasons=collections.Counter();finalized=set();start=time.time();last=0
  # Cache validated checkpoints once on startup; artifacts are immutable after success.
  for t in DIAG:
   for i in range(COUNTS[t]):
    if (R/'folds'/t/f'{i:05d}'/'SUCCESS.json').exists() and valid(t,i):done[t].add(i)
  # Round robin across all five cancers. Every original patient has one job.
  for i in range(max(COUNTS.values())):
   for t,n in COUNTS.items():
    if i<n and i not in done[t]:pending.append((t,i))
  dump(R/'audit/runtime_config.json',dict(signature=sig,initial_workers=capacity(),max_workers=12,threads_per_worker=1,num_gpus=0,raw_sha256=sha(RAW),pid=os.getpid(),expected_total=sum(COUNTS.values()),max_attempts_per_invocation=3,seed=42))
  while pending or active:
   stopping=(R/'STOP').exists()
   for key,(proc,handle,started) in list(active.items()):
    if proc.poll() is None:continue
    handle.close();t,i=key
    if proc.returncode==0 and valid(t,i):done[t].add(i);failed.pop(f'{t}/{i}',None);print('COMPLETED',t,i,'total',sum(map(len,done.values())),flush=True)
    else:
     errors=sorted((R/'folds'/t/f'{i:05d}').glob('failure_*.json'))
     reason=json.loads(errors[-1].read_text())['error'] if errors else f'worker_exit_{proc.returncode}_or_checkpoint_validation_failed'
     reasons[reason.split(':',1)[0]]+=1
     if attempts[key]<3:pending.append(key);print('RETRY',key,'attempt',attempts[key],reason,flush=True)
     else:failed[f'{t}/{i}']=dict(attempts=attempts[key],returncode=proc.returncode,reason=reason)
    del active[key]
   if not stopping:
    limit=capacity()
    while pending and len(active)<limit:
     t,i=pending.popleft();key=(t,i);attempts[key]+=1;log=R/'logs'/t/f'{i:05d}_attempt{attempts[key]}_{time.time_ns()}.log';log.parent.mkdir(exist_ok=True);h=log.open('w')
     proc=subprocess.Popen([sys.executable,'-u',str(SOURCE_ROOT/'src/fold.py'),t,str(i)],stdout=h,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
     active[key]=(proc,h,time.time())
   if time.time()-last>=20 or not active:
    status=dict(status='stopping' if stopping else 'running',pid=os.getpid(),updated_at=time.strftime('%Y-%m-%d %H:%M:%S %z'),elapsed_seconds=time.time()-start,expected_total=8488,completed=sum(map(len,done.values())),failed=len(failed),failure_reason_counts=dict(reasons),pending=len(pending),active=[dict(task=t,fold=i,pid=v[0].pid,elapsed_seconds=time.time()-v[2]) for (t,i),v in active.items()],tasks={t:dict(expected=COUNTS[t],completed=len(done[t]),failed=sum(k.startswith(t+'/') for k in failed)) for t in DIAG},available_memory_GiB=psutil.virtual_memory().available/2**30,load_average=os.getloadavg(),worker_limit=capacity())
    dump(R/'status.json',status);dump(R/'failed_folds.json',failed);last=time.time()
   for t in DIAG:
    if len(done[t])==COUNTS[t] and t not in finalized:
     subprocess.run([sys.executable,str(SOURCE_ROOT/'src/report.py'),'--task',t],check=True);finalized.add(t)
   if stopping and not active:break
   time.sleep(2)
  status['completed']=sum(map(len,done.values()));status['failed']=len(failed);status['status']='stopped' if (R/'STOP').exists() else ('complete' if status['completed']==8488 and not failed else 'incomplete_failed');status['active']=[];dump(R/'status.json',status)
  if status['status']=='complete':
   subprocess.run([sys.executable,str(SOURCE_ROOT/'src/report.py')],check=True)
   print('Completed. Original protected-source checks are documented in audit/; use the repository validator for this portable tree.')
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback
  dump(R/'runner_failure.json',dict(error=traceback.format_exc(),pid=os.getpid()))
  if (R/'status.json').exists():
   st=json.loads((R/'status.json').read_text());st['status']='runner_failed';dump(R/'status.json',st)
  raise
