from pathlib import Path
import hashlib,json,zipfile
root=Path(__file__).parent
files=[p for p in root.iterdir() if p.is_file() and p.suffix in ['.py','.md','.png','.json','.txt','.log']]
# Explicitly include final run logs and the separate exploratory control summary.
files += [root/f'run_{s}.log' for s in [11,22,33]]+[root/f'scale_{s}.log' for s in [11,22,33]]+[root/'scale_summary.json']
files += list((root/'results_final').glob('*.json'))+list((root/'scale_results').glob('*.json'))
files += [root/'data'/n for n in ['tokenizer.json','sources.json','token_counts.json']]
files=sorted(set(files));files=[p for p in files if p.name!='MANIFEST.json']
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));files.append(root/'MANIFEST.json')
out=root.parent/'gqa_E_polskie_benchmarki.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,'gqa_e_benchmark/'+str(p.relative_to(root)))
with zipfile.ZipFile(out) as z:assert z.testzip() is None
print(str(out),out.stat().st_size,len(files))
