#!/usr/bin/env bash
# Reuse immutable inputs; write all new results below a fresh output directory.
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo "Usage: RX_PYTHON=/path/python RX_RUST_PYTHON=/path/python $0 EVIDENCE_ROOT DATASET QUIET_DATASET OUTPUT" >&2
  exit 2
fi
base=$(realpath "$1")
data=$(realpath "$2")
quiet=$(realpath "$3")
out=$(realpath -m "$4")
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
python_bin=${RX_PYTHON:-"$repo/.venv/bin/python"}
rust_python=${RX_RUST_PYTHON:-}
export PYTHONPATH="$repo/src"
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/rx-mpl}
if [[ -e "$out" ]]; then
  echo "Output already exists; refusing to overwrite evidence: $out" >&2
  exit 2
fi
mkdir -p "$out/rx-chain-hardening"
result="$out/rx-chain-hardening"
old="$base/rx-chain-hardening"
original="$base/rx-timing-tracking/final"
for name in parser-replay mac-replay-final stream-frozen stream-fixed-final; do
  ln -s "$old/$name" "$result/$name"
done
"$python_bin" scripts/check_receiver_backend.py --expected-rust no --output "$result/backend-python.json"
"$python_bin" -m pytest tests/ -q --tb=short --cov=nearlink_sdr --cov-fail-under=90.1 \
  --junitxml="$result/python-tests.xml" 2>&1 | tee "$result/python-tests.log"
"$python_bin" -m ruff check src/ tests/
"$python_bin" scripts/rx_chain_crc_audit.py --output "$result/header-crc.json"
"$python_bin" scripts/rx_strict_integration_audit.py --output "$result/strict-integration"
common=(--replay "$old/parser-replay" --original "$original" --dataset "$data" --quiet-dataset "$quiet")
"$python_bin" scripts/rx_stream_frozen_replay.py "${common[@]}" --output "$result/final-stream-python"
"$python_bin" scripts/audit_psk_soft.py --inputs "$old/soft-awgn-python" \
  --legacy-source "$old/soft-baseline/rx_pipeline.py" --output "$result/soft-awgn-python-serial"
"$python_bin" scripts/audit_rx_soft_capture.py --inputs "$old/soft-capture-python" --output "$result/soft-capture-python"
"$python_bin" scripts/audit_rx_boundary.py --replay "$old/parser-replay" \
  --inputs "$old/low-snr-boundary-final" --output "$result/low-snr-boundary-final"
"$python_bin" scripts/audit_rx_boundary.py --replay "$old/parser-replay" \
  --inputs "$result/low-snr-boundary-final" --output "$result/low-snr-boundary-replay"
bench=(--original "$original" --dataset "$data" --sample-id series-02/mode-057/rx-00 \
       --sample-id series-02/mode-065/rx-00 --repeats 5)
"$python_bin" scripts/benchmark_rx_stream.py "${bench[@]}" --output "$result/final-benchmark-python"
if [[ -n "$rust_python" ]]; then
  "$rust_python" scripts/check_receiver_backend.py --expected-rust yes --output "$result/backend-rust.json"
  "$rust_python" -m pytest tests/ -q --tb=short --junitxml="$result/rust-tests.xml" \
    2>&1 | tee "$result/rust-tests.log"
  "$rust_python" scripts/rx_stream_frozen_replay.py "${common[@]}" --output "$result/final-stream-rust"
  "$rust_python" scripts/audit_psk_soft.py --inputs "$old/soft-awgn-python" \
    --legacy-source "$old/soft-baseline/rx_pipeline.py" --output "$result/soft-awgn-rust-serial"
  "$rust_python" scripts/audit_rx_soft_capture.py --inputs "$old/soft-capture-python" --output "$result/soft-capture-rust"
  "$rust_python" scripts/benchmark_rx_stream.py "${bench[@]}" --output "$result/final-benchmark-rust"
  "$python_bin" scripts/summarize_rx_hardening.py --evidence-root "$out" --output "$result/final-report"
else
  echo "Rust comparison not requested: set RX_RUST_PYTHON to an independently built environment."
fi
git rev-parse HEAD > "$result/source-commit.txt"
git diff --binary e2f7b54 HEAD > "$result/source.patch"
sha256sum uv.lock > "$result/lock.sha256"
echo "Evidence: $result"
