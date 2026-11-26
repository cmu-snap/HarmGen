#!/usr/bin/env bash
# Figures 13 and 14 -- compare searches on Reno vs. Westwood.
set -euo pipefail

cd -- "$(dirname -- "$0")"
export MPLBACKEND=Agg
exec python3 section6_compare.py "$@"
