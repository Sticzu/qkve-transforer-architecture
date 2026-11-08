import subprocess,sys,pathlib,concurrent.futures,time
root=pathlib.Path(__file__).parent
# Independent seeds in separate CPU processes; no model/task-specific tuning.
def run(seed):
 with (root/f'run_{seed}.log').open('w') as log:
  p=subprocess.run([sys.executable,'-u',str(root/'bench.py'),'--seeds',str(seed),'--threads','1','--steps-synth','300','--steps-real','600','--out','results_final'],stdout=log,stderr=subprocess.STDOUT)
 return seed,p.returncode
with concurrent.futures.ThreadPoolExecutor(3) as pool:
 for result in pool.map(run,[11,22,33]):print(result,flush=True)
