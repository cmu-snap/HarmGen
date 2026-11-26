#!/usr/bin/env bash
# Figure 12 -- relative harm across five CCA pairs on the BBRv3 VM.
set -euo pipefail

cd -- "$(dirname -- "$0")"
export MPLBACKEND=Agg
exec python3 section6_figure12.py "$@"
