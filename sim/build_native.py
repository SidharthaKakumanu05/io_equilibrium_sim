"""Build the optional exact-operation IO backend: python3 -m sim.build_native.

No fast math: NumPy supplies exp and matmul; C only fuses ordered arithmetic.
The binary and its build manifest must travel with a frozen run snapshot.
"""
from pathlib import Path
import hashlib,json,os,platform,subprocess,sys,sysconfig
import numpy as np

def build():
    here=Path(__file__).resolve().parent
    source=here/'_io_native.c'; binary=here/('_io_native'+sysconfig.get_config_var('EXT_SUFFIX'))
    compiler=os.environ.get('CC','cc')
    command=[compiler,'-shared','-fPIC','-O3','-std=c99','-ffp-contract=off','-fno-fast-math','-fno-associative-math','-I'+sysconfig.get_paths()['include'],'-I'+np.get_include(),str(source),'-o',str(binary)]
    subprocess.run(command,check=True)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    metadata=dict(source_sha256=sha(source),binary_sha256=sha(binary),binary=binary.name,command=command,compiler=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],python=sys.version,numpy=np.__version__,machine=platform.machine())
    (here/'native_build.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))
if __name__=='__main__':build()
