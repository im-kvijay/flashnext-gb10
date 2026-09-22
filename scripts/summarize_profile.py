"""Summarize a torch Chrome trace without treating profiler timing as TPS."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('trace')
p.add_argument('--output',required=True)
a=p.parse_args()
path=Path(a.trace)
opener=gzip.open if path.suffix=='.gz' else open
with opener(path,'rt') as stream:
    data=json.load(stream)
groups=defaultdict(lambda:[0,0.0])
intervals=[]
for event in data.get('traceEvents',[]):
    if event.get('ph')!='X' or event.get('dur',0)<=0:
        continue
    category=event.get('cat','')
    name=event.get('name','')
    if category not in {'kernel','cuda_runtime','gpu_memcpy','gpu_memset','cpu_op'}:
        continue
    groups[(category,name)][0]+=1
    groups[(category,name)][1]+=event['dur']
    if category in {'kernel','gpu_memcpy','gpu_memset'}:
        intervals.append((event['ts'],event['ts']+event['dur']))
merged=[]
for start,end in sorted(intervals):
    if merged and start<=merged[-1][1]:
        merged[-1][1]=max(merged[-1][1],end)
    else:
        merged.append([start,end])
span=(merged[-1][1]-merged[0][0]) if merged else 0
# Attribute idle gaps: which GPU event ended before and started after each gap.
gpu_events=sorted((e for e in data.get('traceEvents',[]) if e.get('ph')=='X' and e.get('dur',0)>0
                   and e.get('cat') in {'kernel','gpu_memcpy','gpu_memset'}),key=lambda e:e['ts'])
gap_groups=defaultdict(lambda:[0,0.0])
end=None; last=None
for event in gpu_events:
    if end is not None and event['ts']-end>200:
        key=f"{last[:60]} -> {event['name'][:60]}"
        gap_groups[key][0]+=1
        gap_groups[key][1]+=event['ts']-end
    if end is None or event['ts']+event['dur']>end:
        end=event['ts']+event['dur']; last=event['name']
busy=sum(end-start for start,end in merged)
report={'trace':str(path),'scope':'profile diagnostic only; durations do not qualify throughput',
        'gpu_event_span_ms':span/1000,'gpu_busy_union_ms':busy/1000,
        'gpu_event_coverage_fraction':busy/span if span else None,
        'idle_gaps_over_200us':sorted(({'between':k,'count':n,'total_ms':d/1000}
                                       for k,(n,d) in gap_groups.items()),key=lambda r:-r['total_ms'])[:15],
        'categories':{}}
for category in sorted({k[0] for k in groups}):
    rows=[{'name':name,'calls':n,'summed_ms':dur/1000,'mean_us':dur/n}
          for (cat,name),(n,dur) in groups.items() if cat==category]
    rows.sort(key=lambda r:r['summed_ms'],reverse=True)
    report['categories'][category]={'calls':sum(r['calls'] for r in rows),
        'summed_ms':sum(r['summed_ms'] for r in rows),'top':rows[:30]}
Path(a.output).write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='categories'},indent=2))
for row in report['idle_gaps_over_200us'][:6]:
    print(f"idle {row['total_ms']:.1f} ms in {row['count']} gaps: {row['between']}")
for row in report['categories'].get('kernel',{}).get('top',[])[:12]:
    print(f"{row['summed_ms']:.3f} ms, {row['calls']} calls: {row['name'][:150]}")
