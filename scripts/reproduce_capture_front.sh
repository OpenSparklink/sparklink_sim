#!/usr/bin/env bash
# Reuse hash-frozen inputs; never overwrite original evidence or IQ.
set -euo pipefail
if [ "$#" -ne 4 ]; then
  echo 'usage: reproduce_capture_front.sh EVIDENCE_ROOT IQ_DATASET QUIET_DATASET NEW_OUTPUT' >&2
  exit 2
fi
baseline_root=$1
iq_dataset=$2
quiet_dataset=$3
new_output=$4
mkdir "$new_output"
export PYTHONPATH=src MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/rx-mpl} UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/rx-uv}
run_python() { uv run --frozen --no-sync python "$@"; }
run_python scripts/evaluate_capture_front.py --split development --inputs "$baseline_root/rx-capture-front/development-screened" --output "$new_output/development-screened"
run_python scripts/evaluate_capture_front.py --split held-out --inputs "$baseline_root/rx-capture-front/held-out/inputs" --check-chunks --output "$new_output/held-out"
run_python scripts/audit_capture_negatives.py --output "$new_output/independent-negatives"
run_python scripts/rx_stream_frozen_replay.py --replay "$baseline_root/rx-chain-hardening/parser-replay" --original "$baseline_root/rx-timing-tracking/final" --dataset "$iq_dataset" --quiet-dataset "$quiet_dataset" --output "$new_output/original-frozen"
run_python scripts/audit_rx_boundary.py --replay "$baseline_root/rx-chain-hardening/parser-replay" --inputs "$baseline_root/rx-chain-hardening/low-snr-boundary-final" --output "$new_output/uncoded-boundary"
run_python scripts/audit_rx_soft_capture.py --inputs "$baseline_root/rx-chain-hardening/soft-capture-python" --output "$new_output/coded-replay"
if [ -n "${RX_RUST_PYTHON:-}" ]; then
  "$RX_RUST_PYTHON" -c 'from nearlink_sdr.common.polar import _HAS_RUST_ACCEL; assert _HAS_RUST_ACCEL'
  "$RX_RUST_PYTHON" scripts/evaluate_capture_front.py --split held-out --inputs "$baseline_root/rx-capture-front/held-out/inputs" --check-chunks --output "$new_output/held-out-rust"
  "$RX_RUST_PYTHON" scripts/audit_rx_soft_capture.py --inputs "$baseline_root/rx-chain-hardening/soft-capture-python" --output "$new_output/coded-replay-rust"
  run_python scripts/summarize_capture_front.py --evidence-root "$baseline_root" --capture-root "$new_output"
else
  echo 'Rust comparison explicitly skipped: set RX_RUST_PYTHON to an installed accelerated interpreter.' >&2
fi
