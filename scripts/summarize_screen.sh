#!/bin/bash
# Print the key numbers from a run_screen result directory.
d=$1
cd "$d" || exit 1
cat screen-results.json 2>/dev/null | tr -d '\n '; echo
for f in retrieval-c8-4k throughput-c8-4k workload-c8-4k workload-c8-4k-repeat; do
  [ -f $f.json ] && python3 -c "
import json;d=json.load(open('$f.json'));s=d['summary'];c=d.get('engine_counter_deltas',{})
acc=c.get('vllm:spec_decode_num_accepted_tokens_total');dr=c.get('vllm:spec_decode_num_drafts_total')
print('$f', 'overlap_tps=%.1f'%(s['all_streams_overlap_output_tps'] or 0), 'e2e_tps=%.1f'%s['aggregate_output_tps_including_prefill'],
 'errors',s['errors'],'natural',s['naturally_finished_requests'],'correct',s['retrieval_correct'],
 'mean_accept_len=%.2f'%(1+acc/dr) if acc and dr else '')"
done
[ -f retention.json ] && python3 -c "import json;d=json.load(open('retention.json'));print('retention',{k:v for k,v in d.get('summary',d).items() if not isinstance(v,(list,dict))})"
[ -f tools-c8-4k.json ] && python3 -c "import json;d=json.load(open('tools-c8-4k.json'));print('tools',{k:v for k,v in d.get('summary',d).items() if not isinstance(v,(list,dict))})"
