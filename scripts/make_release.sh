#!/usr/bin/env bash
# Build a self-contained release bundle: this repository at HEAD plus the files git does not carry.
#   bash scripts/make_release.sh --drafter mtp_trained.pt [--drafter-report report.json] \
#        --fidelity-source profile-workload-c8-4k.json --fidelity-dense fidelity.json \
#        --fidelity-profile fidelity.json [--out dist]
# Writes <out>/flashnext-gb10-<commit>.tar.gz. On the target GB10:
#   tar xzf flashnext-gb10-<commit>.tar.gz && cd flashnext-gb10-<commit> && bash scripts/setup_gb10.sh
# setup_gb10.sh picks up release/drafter/mtp_trained.pt automatically. Refuses a dirty tree so the
# bundle matches a commit.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
OUT=dist
declare -A SRC
while (($#)); do
  case $1 in
    --drafter) SRC[drafter/mtp_trained.pt]=$2; shift 2;;
    --drafter-report) SRC[drafter/report.json]=$2; shift 2;;
    --fidelity-source) SRC[reference/fidelity-source.json]=$2; shift 2;;
    --fidelity-dense) SRC[reference/fidelity-bf16-dense.json]=$2; shift 2;;
    --fidelity-profile) SRC[reference/fidelity-profile.json]=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[[ -n ${SRC[drafter/mtp_trained.pt]:-} ]] || { echo "--drafter is required" >&2; exit 2; }
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo "commit your changes first" >&2; exit 1; }
NAME=flashnext-gb10-$(git rev-parse --short=10 HEAD)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
git archive --prefix="$NAME/" HEAD | tar -x -C "$STAGE"
for dst in "${!SRC[@]}"; do
  mkdir -p "$STAGE/$NAME/release/$(dirname "$dst")"
  cp "${SRC[$dst]}" "$STAGE/$NAME/release/$dst"
done
(cd "$STAGE/$NAME/release" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
mkdir -p "$OUT"
tar -C "$STAGE" -czf "$OUT/$NAME.tar.gz" "$NAME"
sha256sum "$OUT/$NAME.tar.gz" | tee "$OUT/$NAME.tar.gz.sha256"
cat "$STAGE/$NAME/release/SHA256SUMS"
