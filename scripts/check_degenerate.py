"""Flag degenerate generations in bench/concurrency.py results (state-corruption canary).

A corrupted recurrent state shows up as a stream collapsing into one repeated
token ("Register Register ...") or digits, while throughput still looks fine.
For each natural-stop or workload request, the tail of the text is scored by
its zlib compression ratio, its most common word share and runs of 200+ digits. Performance-mode
runs (forced continuation past EOS) repeat legitimately and are skipped.

Usage: check_degenerate.py RESULT.json [...]; exit status 1 if any stream is degenerate.
"""
import json
import re
import sys
import zlib
from collections import Counter


def degenerate(text, tail=3000):
    t = text[-tail:]
    if len(t) < 500:
        return False, 0.0, 0.0
    ratio = len(zlib.compress(t.encode())) / len(t.encode())
    words = t.split()
    share = Counter(words).most_common(1)[0][1] / len(words) if words else 0.0
    digits = max((len(m) for m in re.findall(r'\d{200,}', t)), default=0)  # e.g. '000000...' or looping digits
    return ratio < 0.12 or share > 0.5 or digits > 0, ratio, share


def main():
    bad = 0
    for path in sys.argv[1:]:
        d = json.load(open(path))
        if 'performance' in path or 'throughput' in path:
            continue
        for r in d['requests']:
            flag, ratio, share = degenerate(r.get('text') or '')
            if flag:
                bad += 1
                print(f'{path}: agent {r["agent"]} degenerate (zlib {ratio:.2f}, top word {share:.0%}): '
                      f'{(r.get("text") or "")[-120:]!r}')
    print('degenerate streams:', bad)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
