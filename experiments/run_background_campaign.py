"""Resumable background-only campaign. Use frozen specs; never overwrite old runs."""
import argparse
import os
from pathlib import Path
import sys
# Set before importing NumPy so each worker consumes a single CPU thread.
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['MPLBACKEND']='Agg'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','validate','analyze','supervise','launch'])
    p.add_argument('--spec');p.add_argument('--out');p.add_argument('--analysis');p.add_argument('--campaign')
    p.add_argument('--resume',action='store_true');p.add_argument('--fault');p.add_argument('--fault-step',type=int)
    p.add_argument('--stop-step',type=int)
    a=p.parse_args()
    if a.mode=='run':
        from experiments.pipeline.runner import run
        run(a.spec,a.out,a.resume,a.fault,a.fault_step,a.stop_step)
        if a.analysis:
            from experiments.pipeline.analyze import analyze
            analyze(a.out,a.analysis)
    elif a.mode=='validate':
        from experiments.pipeline.validate import validate
        print(validate(a.out))
    elif a.mode=='analyze':
        from experiments.pipeline.analyze import analyze
        analyze(a.out,a.analysis)
    else:
        from experiments.pipeline.supervisor import launch,supervise
        (launch if a.mode=='launch' else supervise)(a.campaign,a.resume)
if __name__=='__main__':main()
