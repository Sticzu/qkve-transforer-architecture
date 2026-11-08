import subprocess,sys,pathlib,concurrent.futures
root=pathlib.Path(__file__).parent
def run(seed):
 with (root/f'scale_{seed}.log').open('w') as log:
  p=subprocess.run([sys.executable,'-u',str(root/'bench.py'),'--tasks','wiki,books','--variants','scaled_matched','--seeds',str(seed),'--threads','1','--steps-real','600','--out','scale_results'],stdout=log,stderr=subprocess.STDOUT)
 return seed,p.returncode
with concurrent.futures.ThreadPoolExecutor(3) as pool:
 for r in pool.map(run,[11,22,33]):print(r,flush=True)
