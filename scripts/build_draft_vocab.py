"""Build a draft-only vocabulary from a named public code/docs directory.

Do not point this at company code, benchmark solutions, or evaluation outputs.
The target model and its full vocabulary remain unchanged.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

from transformers import AutoTokenizer

p=argparse.ArgumentParser()
p.add_argument('--model',required=True)
p.add_argument('--corpus',required=True)
p.add_argument('--output',required=True)
p.add_argument('--size',type=int,default=65536)
p.add_argument('--max-bytes',type=int,default=32*1024**2)
p.add_argument('--corpus-revision',help='Source commit when using an exported tree')
a=p.parse_args()
tokenizer=AutoTokenizer.from_pretrained(a.model)
root=Path(a.corpus).resolve()
counts=Counter()
sources=[]
total=0
for path in sorted(root.rglob('*')):
    if path.suffix not in {'.py','.md','.cu','.c','.h','.cpp'} or not path.is_file():
        continue
    relative=path.relative_to(root)
    if any(part in {'tests','test','benchmarks','evals','data','.git'} for part in relative.parts):
        continue
    blob=path.read_bytes()
    if total+len(blob)>a.max_bytes:
        break
    text=blob.decode('utf-8',errors='replace')
    counts.update(tokenizer.encode(text,add_special_tokens=False,verbose=False))
    sources.append({'path':str(relative),'sha256':hashlib.sha256(blob).hexdigest()})
    total+=len(blob)
if not counts:
    raise SystemExit('no corpus tokens')
keep=set(tokenizer.all_special_ids)
for token,_ in counts.most_common():
    if len(keep)>=a.size:
        break
    keep.add(token)
revision=a.corpus_revision or subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
out={'model_revision':'fc694b54fb0174e0913e6adf86691ef85a4ead47',
     'corpus_revision':revision,'corpus_bytes':total,'source_files':sources,
     'token_ids':sorted(keep),'total_token_occurrences':counts.total(),
     'corpus_coverage':sum(counts[t] for t in keep)/counts.total()}
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ['corpus_revision','corpus_bytes','total_token_occurrences','corpus_coverage']}))
print('Draft vocabulary size:',len(keep))
