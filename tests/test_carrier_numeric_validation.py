"""A passing packet label cannot waive node quantization/phase budgets."""

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nearlink_sdr.phy.carrier_weights import bounded_carrier_weights
from nearlink_sdr.phy.mac_interface import MacStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from tests.test_strict_receive import iq_record, transmitted


@pytest.fixture
def numeric_fixture(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("audit_carrier_weight_numeric")
    source = tmp_path / "iq.npz"
    np.savez(source, iq=np.ones(40))
    meta = dict(id="fixed", path=str(source), sha256=module.sha(source))
    directories = [tmp_path / "float", tmp_path / "fixed"]
    for directory, fixed in zip(directories, [False, True], strict=True):
        _, info = bounded_carrier_weights(np.ones(40), 2, fixed=fixed)
        row = dict(input=meta, frames=[dict(correct=True, status="valid", modulation="QPSK",
                   data_start_symbol=0, frame_end_symbol=40, carrier=dict(weights=info,
                   residual_cycles_per_symbol=0., residual_phase_radians=0.))])
        (directory / "evidence").mkdir(parents=True)
        np.savez(directory / "evidence/fixed.npz", **{"0-symbols": np.ones(40, complex)})
        (directory / "results.json").write_text(json.dumps([row]))
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps(dict(numerical_budget=dict(
        weight_max_difference_over_max_reference=.002,
        carrier_frequency_cycles_per_symbol_difference=1e-5,
        carrier_phase_difference_radians=.005))))
    return module, SimpleNamespace(before=directories[0], after=directories[1], budget=budget,
                                   output=tmp_path / "report")


def test_matching_labels_and_nodes_pass_only_the_stated_numeric_gate(numeric_fixture):
    module, args = numeric_fixture
    assert module.run(args) == 0
    report = json.loads((args.output / "numeric.json").read_text())
    assert report["numeric_pass"] and not report["whole_system_accepted"]


def test_same_packet_label_cannot_waive_phase_error(numeric_fixture):
    module, args = numeric_fixture
    rows = json.loads((args.after / "results.json").read_text())
    rows[0]["frames"][0]["carrier"]["residual_phase_radians"] = .01
    (args.after / "results.json").write_text(json.dumps(rows))
    assert module.run(args) == 1
    report = json.loads((args.output / "numeric.json").read_text())
    assert not report["gates"]["phase"]
    assert report["records"][0]["correct_pair"] == [True, True]


def test_same_packet_label_cannot_waive_weight_node_error(numeric_fixture, monkeypatch):
    module, args = numeric_fixture
    original = module.bounded_carrier_weights

    def corrupted(power, kappa, *, fixed=False):
        result, info = original(power, kappa, fixed=fixed)
        return (2 * result if fixed else result), info

    monkeypatch.setattr(module, "bounded_carrier_weights", corrupted)
    assert module.run(args) == 1
    assert not json.loads((args.output / "numeric.json").read_text())["gates"]["weight"]


def test_numeric_replay_rejects_replaced_input_metadata(numeric_fixture):
    module, args = numeric_fixture
    rows = json.loads((args.after / "results.json").read_text())
    changed = copy.deepcopy(rows)
    changed[0]["input"]["sha256"] = "0" * 64
    (args.after / "results.json").write_text(json.dumps(changed))
    with pytest.raises(AssertionError):
        module.run(args)


def test_no_evaluated_nodes_cannot_pass_numeric_acceptance(numeric_fixture):
    module, args = numeric_fixture
    for directory in [args.before, args.after]:
        (directory / "results.json").write_text("[]")
    assert module.run(args) == 1
    assert not json.loads((args.output / "numeric.json").read_text())["numeric_pass"]


def test_packet_fft_workload_keeps_capacity_and_system_failures(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("model_carrier_workload")
    output = tmp_path / "workload"
    assert module.run(SimpleNamespace(output=output)) == 1
    report = json.loads((output / "workload.json").read_text())
    assert not report["whole_system_accepted"] and not report["format_complex32_validated"]
    rows = {r["mcs"]: r for r in report["cases"] if r["payload_bytes"] == 2047}
    assert rows[12]["fft_points"] == 131072
    assert rows[12]["bram36_capacity_lower_tiles"] == 114
    assert rows[8]["fft_points"] == 262144
    assert not rows[8]["fits_entire_chip_bram_capacity"]
    assert not rows[0]["fits_entire_chip_bram_capacity"]


def test_reproducer_rejects_changed_frozen_budget_before_outputs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("reproduce_carrier_weights")
    (tmp_path / "predevelopment.json").write_text("{}")
    with pytest.raises(AssertionError):
        module.run(SimpleNamespace(evidence=tmp_path, output=tmp_path / "not-created"))
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize("damage", ["trace", "decision"])
def test_reproducer_checks_decisions_and_trace_independently(tmp_path, monkeypatch, damage):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("reproduce_carrier_weights")
    row = dict(input=dict(id="sample"), correct=1, timing_backend="python", integer_backend="numpy")
    for arm in ["a", "b"]:
        directory = tmp_path / arm
        (directory / "evidence").mkdir(parents=True)
        (directory / "results.json").write_text(json.dumps([row]))
        np.savez(directory / "evidence/sample.npz", centers=np.arange(4))
    assert module.identity(tmp_path / "a", tmp_path / "b")["exact"]
    if damage == "trace":
        np.savez(tmp_path / "b/evidence/sample.npz", centers=np.arange(4) + .01)
    else:
        row["correct"] = 0
        (tmp_path / "b/results.json").write_text(json.dumps([row]))
    with pytest.raises(AssertionError):
        module.identity(tmp_path / "a", tmp_path / "b")


@pytest.mark.parametrize("ft,mcs", [(2, 8), (2, 12), (3, 8), (3, 12), (4, 8), (4, 12)])
@pytest.mark.parametrize("length", [1, 2047])
def test_selected_weights_at_legal_psk_length_extremes(ft, mcs, length):
    cfg = TxConfig(frame_type=ft, mcs_index=mcs, symbol_rate_mhz=4, pilot_interval=8,
                   whitening_seed=0x52, crc_seed=0x123456, ctrl_bits_len=28 if ft == 2 else 27,
                   head_llid=None if ft == 2 else 0x234567)
    # Preserve the original payload-only B1 waveform and 2047-byte stress input.
    # Its historical test name is not a standard B1 legal-length claim.
    profile = PSKReceiveProfile.ordinary(
        cfg, length_contract="standard" if ft == 2 else "legacy-b1-payload-length")
    wire, wanted = transmitted(profile, length=length)
    x, quiet, _, _ = iq_record(wire, cfo=23000, ppm=100)
    for fixed in [False, True]:
        rx = MacStreamReceiver(profile, quiet, carrier_weight_kappa=2, carrier_weight_fixed=fixed)
        packets = []
        for start in range(0, len(x), 997):
            packets += rx.feed(x[start:start + 997]).packets
        packets += rx.finish().packets
        assert [packet.mac_payload for packet in packets] == [wanted]
