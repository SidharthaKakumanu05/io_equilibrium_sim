"""Standalone durable queue, conservative concurrency, and durable progress updates."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from experiments.pipeline.common import atomic_json,check_space,digest,identity,lock,utc

THREAD_ENV=dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1')

def available_run_index(remaining, children):
    """Keep each frozen CPU exclusive, even when workers finish out of order."""
    occupied={run['cpu'] for _,run,_ in children.values()}
    return next((i for i,run in enumerate(remaining) if run['cpu'] not in occupied),None)

def command(campaign,run,mode='run'):
    c=[sys.executable,str(Path(campaign['snapshot'])/'experiments/run_background_campaign.py'),mode]
    if mode=='run':c+=['--spec',run['spec'],'--out',run['out'],'--analysis',run['analysis']]
    elif mode=='analyze':c+=['--out',run['out'],'--analysis',run['analysis']]
    return c

def monitor_progress(campaign,super_ident,status,blocker=None):
    p=Path(campaign['progress_path'])
    x=json.loads(p.read_text());x['updated_at_utc']=utc()
    runs=[]
    for r in campaign['runs']:
        rr=dict(r);s=Path(r['out'])/'status.json'
        if s.exists():rr.update(json.loads(s.read_text()))
        else:rr['status']='queued'
        resume_cmd=command(campaign,r,'analyze') if (Path(r['out'])/'COMPLETE').exists() else command(campaign,r)+(['--resume'] if Path(r['out']).exists() else [])
        rr['resume_command']='cd '+campaign['snapshot']+' && '+__import__('shlex').join(resume_cmd)
        rr['analysis_resume_command']='cd '+campaign['snapshot']+' && '+__import__('shlex').join(command(campaign,r,'analyze'))
        a=Path(r['analysis'])/'ANALYSIS_COMPLETE.json';rr['analysis_complete']=a.exists()
        if (Path(r['out'])/'CURRENT.json').exists():rr['current_checkpoint']=json.loads((Path(r['out'])/'CURRENT.json').read_text())
        runs.append(rr)
    x['runs']=runs;x['run_resume_commands']=[r['resume_command'] for r in runs]
    stage=x['stages']['experiment_runner'];stage['supervisor_identity']=super_ident;stage['campaign_status']=status
    stage['campaign_manifest']=campaign['manifest_path'];stage['status']='complete' if status=='complete' else 'in_progress'
    if blocker:x['blockers']=[dict(scope='campaign',reason=blocker)]
    elif status=='complete':
        x['blockers']=[];x['active_stage']='successful_completion';x['source_edit_owner']=None
        stage['completed_at_utc']=utc()
        if 'experiment_runner' not in x['completed_stages']:x['completed_stages'].append('experiment_runner')
        x['stages']['successful_completion'].update(status='ready_for_coordinator_audit',evidence=[r['out']+'/COMPLETE' for r in runs])
    else:x['blockers']=[]
    atomic_json(p,x)

def supervise(path,resume=False):
    path=Path(path);c=json.loads(path.read_text());root=Path(c['storage_root'])
    os.sched_setaffinity(0,{c['runs'][0]['cpu']})
    guard=lock(root/'CAMPAIGN.lock');ident=identity();children={};remaining=list(c['runs']);stop=[]
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.append(True))
    atomic_json(root/'supervisor_status.json',dict(status='running',identity=ident,started=utc()))
    try:
        while remaining or children:
            if stop:raise InterruptedError('Supervisor termination requested')
            check_space(root,c['policy'])
            while remaining and len(children)<c['concurrency']:
                index=available_run_index(remaining,children)
                if index is None:break
                r=remaining.pop(index);out=Path(r['out'])
                if (out/'COMPLETE').exists():
                    if not (Path(r['analysis'])/'ANALYSIS_COMPLETE.json').exists():cmd=command(c,r,'analyze')
                    else:
                        from experiments.pipeline.validate import validate
                        validate(out)
                        marker=json.loads((Path(r['analysis'])/'ANALYSIS_COMPLETE.json').read_text())
                        for fn,key in [('summary.json','summary_sha256'),('overview.png','figure_sha256')]:
                            if digest(Path(r['analysis'])/fn)!=marker[key]:raise ValueError('Analysis checksum mismatch')
                        if digest(out/'COMPLETE')!=marker['raw_complete_sha256']:raise ValueError('Analysis provenance mismatch')
                        continue
                else:
                    cmd=command(c,r)
                    if out.exists():
                        test_guard=lock(out/'RUN.lock');test_guard.close() # live run stops supervisor; never duplicate
                        if not resume:raise ValueError('Existing partial run requires explicit --resume: '+str(out))
                        old=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
                        if old.get('status')=='failed':raise ValueError('Failed run requires investigation before resume: '+str(out))
                        cmd+=['--resume']
                logfile=open(root/(r['id']+'-'+str(time.time_ns())+'.log'),'xb')
                proc=subprocess.Popen(cmd,cwd=c['snapshot'],env={**os.environ,**THREAD_ENV},stdin=subprocess.DEVNULL,stdout=logfile,stderr=subprocess.STDOUT)
                children[proc.pid]=(proc,r,logfile)
                monitor_progress(c,ident,'running')
            for pid,(proc,r,logfile) in list(children.items()):
                code=proc.poll()
                if code is not None:
                    logfile.close();del children[pid]
                    if code!=0:raise RuntimeError(f"Run {r['id']} exited {code}; campaign stopped")
                    if not (Path(r['out'])/'COMPLETE').exists() or not (Path(r['analysis'])/'ANALYSIS_COMPLETE.json').exists():
                        raise ValueError('Missing success markers: '+r['id'])
            monitor_progress(c,ident,'running')
            if remaining or children:time.sleep(30)
        monitor_progress(c,ident,'complete')
        atomic_json(root/'CAMPAIGN_COMPLETE.json',dict(status='passed',utc=utc(),identity=ident,runs=[r['id'] for r in c['runs']]))
        atomic_json(root/'supervisor_status.json',dict(status='complete',identity=ident,utc=utc()))
    except BaseException as exc:
        for proc,_,_ in children.values():
            if proc.poll() is None:proc.terminate()
        for proc,_,logfile in children.values():
            proc.wait();logfile.close()
        monitor_progress(c,ident,'stopped',repr(exc))
        atomic_json(root/'supervisor_status.json',dict(status='stopped',error=repr(exc),identity=ident,utc=utc()))
        raise
    finally:guard.close()

def launch(path,resume=False):
    path=Path(path).resolve();c=json.loads(path.read_text());root=Path(c['storage_root'])
    if 'sandbox' in Path('/proc/1/cmdline').read_bytes().decode(errors='replace'):
        raise RuntimeError('Durable launch requires host process namespace, not tool sandbox')
    test_guard=lock(root/'CAMPAIGN.lock');test_guard.close()
    if not resume and (root/'supervisor_status.json').exists():raise ValueError('Supervisor already started; inspect before resume')
    env={**os.environ,**THREAD_ENV}
    cmd=[sys.executable,str(Path(c['snapshot'])/'experiments/run_background_campaign.py'),'supervise','--campaign',str(path)]
    if resume:cmd+=['--resume']
    logfile=open(root/('supervisor-'+str(time.time_ns())+'.log'),'xb')
    child=subprocess.Popen(cmd,cwd=c['snapshot'],env=env,stdin=subprocess.DEVNULL,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
    logfile.close()
    atomic_json(root/'launch.json',dict(pid=child.pid,command=cmd,utc=utc(),launcher=identity()))
    print(json.dumps(dict(pid=child.pid,campaign=str(path),command=cmd)),flush=True)
