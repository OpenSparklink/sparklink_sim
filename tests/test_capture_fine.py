"""Variable actual geometry,causal source,exact ownership and retention failures."""

import hashlib
import importlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.capture_fine import CaptureFineWorker, pack_filter_word, unpack_filter_word
from nearlink_sdr.sim.capture_transport import (
    CaptureRequest,
    CaptureTransport,
    CaptureTransportAssumptions,
    FineService,
)
from nearlink_sdr.sim.spline_score import SplineScoreAssumptions, SplineScoreFlow, SyncSignTemplate


def archive_reference(path):
    """Independent exact NPY-member reference;container bytes are separate."""
    with zipfile.ZipFile(path) as archive:
        members = {
            name: dict(
                sha256=hashlib.sha256(archive.read(name)).hexdigest(), bytes=len(archive.read(name))
            )
            for name in archive.namelist()
        }
    return dict(
        original_archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), members=members
    )


@pytest.fixture
def input_container_verifier(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("model_capture_fine").verify_input_container


def test_repacked_container_requires_exact_original_members_and_explicit_derived_mode(
    tmp_path, input_container_verifier
):
    original, derived = tmp_path / "original.npz", tmp_path / "derived.npz"
    np.savez_compressed(
        original,
        iq=np.array([1 + 2j, -0.0 + 0j]),
        quiet=np.zeros(3),
        payload=np.array([3, 255], np.uint8),
    )
    expected = archive_reference(original)
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(derived, "w") as target:
        for name in reversed(source.namelist()):
            target.writestr(name, source.read(name), compress_type=zipfile.ZIP_STORED)
    exact = input_container_verifier(original, expected, require_original=True)
    assert exact["original_archive_hash_matches"]
    result = input_container_verifier(derived, expected, require_original=False)
    assert result["exact_original_member_hashes_verified"]
    assert result["original_archive_sha256"] == expected["original_archive_sha256"]
    assert result["current_archive_sha256"] == hashlib.sha256(derived.read_bytes()).hexdigest()
    assert not result["original_archive_hash_matches"]
    assert result["members"] == exact["members"]
    with pytest.raises(ValueError, match="Frozen original archive SHA mismatch"):
        input_container_verifier(derived, expected, require_original=True)


@pytest.mark.parametrize(
    "change", ["one_ulp", "payload_bit", "header", "missing", "extra", "duplicate", "size"]
)
def test_derived_mode_rejects_data_header_and_membership_changes(
    tmp_path, input_container_verifier, change
):
    original, derived = tmp_path / "original.npz", tmp_path / "derived.npz"
    np.savez_compressed(original, iq=np.array([1.0, 2.0]), payload=np.array([3], np.uint8))
    expected = archive_reference(original)
    with zipfile.ZipFile(original) as source:
        members = {name: source.read(name) for name in source.namelist()}
    if change in ("one_ulp", "payload_bit"):
        name = "iq.npy" if change == "one_ulp" else "payload.npy"
        values = np.load(io.BytesIO(members[name]), allow_pickle=False)
        if change == "one_ulp":
            values[0] = np.nextafter(values[0], np.inf)
        else:
            values[0] ^= 1
        buffer = io.BytesIO()
        np.save(buffer, values, allow_pickle=False)
        members[name] = buffer.getvalue()
    elif change == "header":
        # Same numerical values;an altered NPY header is still not the original byte reference.
        members["iq.npy"] = members["iq.npy"].replace(
            b"'fortran_order': False", b"'fortran_order': True ", 1
        )
    elif change == "missing":
        del members["payload.npy"]
    elif change == "extra":
        members["extra.npy"] = members["iq.npy"]
    elif change == "size":
        members["iq.npy"] += b"\0"
    with zipfile.ZipFile(derived, "w") as target:
        for name, value in members.items():
            target.writestr(name, value)
        if change == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                target.writestr("iq.npy", members["iq.npy"])
    with pytest.raises(ValueError, match=r"Input (member|archive)"):
        input_container_verifier(derived, expected, require_original=False)


def word_at(index, path):
    return ((index * 104729) ^ (path * 0x15249)) & ((1 << 46) - 1)


class CausalSource:
    """Generates each next word on demand,not an allocated capture oracle."""

    def __init__(self, n):
        self.n, self.seen = n, 0

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        assert type(index) is int and index == self.seen
        self.seen += 1
        return (word_at(index, 0), word_at(index, 1))


@pytest.mark.parametrize("window,history", [(256, 384), (376, 504), (632, 760)])
def test_actual_geometries_copy_exact_values_on_causal_source(window, history):
    source = CausalSource(window + 641)
    a = CaptureTransportAssumptions(window=window, history=history)
    requests = [CaptureRequest(i, i % 2, (2, 6)) for i in range(window, len(source) + 1, 128)]
    events = []

    def worker(event, words):
        assert words == [
            word_at(i, event["path"])
            for i in range(event["after_samples"] - window, event["after_samples"])
        ]
        return FineService(17, "ready", 16)

    report = CaptureTransport(a).simulate(
        source, requests, fine_worker=worker, sink=lambda e, w: events.append(e)
    )
    assert source.seen == len(source) and report["completed"] == len(requests)
    assert all(e["retention_necessary_pass"] for e in events)
    cost = CaptureTransport(a).estimate(measured_worker=True)
    assert cost["descriptor_reserved_bits_each"] == 448
    assert sum(cost["descriptor_fields_bits"].values()) <= 448
    assert cost["register_reservation_bits"]["read_pipeline"] == 2 * (
        46 + (window - 1).bit_length() + 4 + 1
    )
    assert cost["transport_bram36"] == 10
    assert not report["replay_read_ports_scheduled"] and not report["handoff_check_latency_modeled"]


def test_retained_snapshot_cannot_hide_overwritten_post_sync_continuation():
    a = CaptureTransportAssumptions(clock_hz=100000000, history=256)
    events = []

    def worker(e, w):
        return FineService(5000, "ready", 8)

    report = CaptureTransport(a).simulate(
        CausalSource(2048),
        [CaptureRequest(256, 0, (2,))],
        fine_worker=worker,
        sink=lambda e, w: events.append(e),
    )
    assert report["completed"] == 1 and report["worker_results"]["ready_retention_lost"] == 1
    event = events[0]
    assert (
        event["continuation_gap_samples"]
        == max(0, event["continuous_count_after_edge"] - 256 - 256)
        > 0
    )
    assert not event["retention_necessary_pass"]
    # EOF cannot create further overwrites,even while this slow worker drains.
    eof = []
    CaptureTransport(a).simulate(
        CausalSource(280),
        [CaptureRequest(256, 0, (2,))],
        fine_worker=worker,
        sink=lambda e, w: eof.append(e),
    )
    assert eof[0]["continuous_count_after_edge"] == 280 and eof[0]["retention_necessary_pass"]
    assert eof[0]["cycle"] > 280 * a.clock_hz // a.input_hz


def test_worker_waits_rejections_and_resource_drops_are_separate():
    dispositions = [
        ("needs_future", 16),
        ("below_threshold", 16),
        ("no_candidate", None),
        ("numerical_reject", None),
    ]
    counts = {}

    def worker(event, words):
        kind, tick = dispositions[event["id"]]
        counts[event["id"]] = 1
        return FineService(10, kind, tick)

    result = CaptureTransport().simulate(
        CausalSource(4096),
        [CaptureRequest(i, 0, (2,)) for i in (256, 1024, 2048, 3072)],
        fine_worker=worker,
    )
    assert len(counts) == 4 and result["completed"] == 4
    assert all(result["worker_results"][kind] == 1 for kind, _ in dispositions)
    assert result["worker_results"]["ready"] == 0
    blocked = []
    r = CaptureTransport().simulate(
        CausalSource(2048),
        [CaptureRequest(i, 0, (2,)) for i in range(256, 2049, 32)],
        fine_worker=lambda e, w: FineService(1000, "ready", 8),
        stalls=((0, 40000),),
        sink=lambda e, w: blocked.append(e),
    )
    assert r["queue_drops"] > 0 and any(
        e["status"] == "owned_queue_full" and e["disposition"] is None for e in blocked
    )
    assert r["input_samples"] == 2048 and not r["resource_feedback_to_capture_algorithm"]


@pytest.mark.parametrize("ft,count,window", [(2, 32, 256), (3, 62, 376), (4, 126, 632)])
def test_actual_profile_worker_and_microtrace_values_match_direct_node(ft, count, window):
    cfg = TxConfig(
        frame_type=ft,
        mcs_index=8,
        pid=3,
        symbol_rate_mhz=4,
        ctrl_bits_len=28 if ft == 2 else 27,
        head_llid=None if ft == 2 else 0x234567,
    )
    profile = PSKReceiveProfile.ordinary(cfg)
    sync = profile.sync_symbols()
    assert len(sync) == count
    template = SyncSignTemplate.from_symbols(sync)
    # Independent linear construction,integer sample locations contain exact template values.
    times = 32 + 4 * np.arange(count)
    samples = np.interp(np.arange(window), times, sync.real) + 1j * np.interp(
        np.arange(window), times, sync.imag
    )
    triples = [(round(z.real * 1024), round(z.imag * 1024), 0) for z in samples]
    packed = [pack_filter_word(*w) for w in triples]
    request = CaptureRequest(window, 0, (32,), 8, 128)
    results = []
    worker = CaptureFineWorker(
        template, window, confirmation_required=False, sink=lambda e, r: results.append(r)
    )
    completed = []
    source = [(w, w) for w in packed] + [(0, 0)] * 128
    transport = CaptureTransport(CaptureTransportAssumptions(window=window, history=window + 128))
    report = transport.simulate(
        source, [request], fine_worker=worker, sink=lambda e, w: completed.append(e)
    )
    assert report["completed"] == 1 and completed[0]["retention_necessary_pass"]
    service = FineService(
        completed[0]["fine_cycles"], completed[0]["disposition"], completed[0]["start_tick"]
    )
    direct = SplineScoreFlow(SplineScoreAssumptions(capacity=window)).run(
        triples, template, (32,), minimum_tick=8, lookahead_ticks=128
    )
    assert results == [direct] and service.cycles == direct["cycles"]
    assert service.disposition == "ready" and service.start_tick == 256
    estimate = worker.estimate()
    assert estimate["raw_source_is_same_owned_slot"] and estimate["raw_read_ports"] == 1
    assert (
        estimate["preparation_DSP25x18_estimate"] == 2
        and estimate["scoring_DSP25x18_estimate"] == 1
    )
    assert not estimate["multiplier_sharing_credit"] and not estimate["complete_receiver_verified"]


def test_word_serialization_extremes_and_bad_fields_never_get_wrapped():
    for triple in [(-(2**18), 2**18 - 1, -128), (0, -1, 127), (1, 0, 0)]:
        assert unpack_filter_word(pack_filter_word(*triple)) == triple
    for triple in [(True, 0, 0), (np.bool_(True), 0, 0), (0.0, 0, 0), (2**18, 0, 0), (0, 0, 128)]:
        with pytest.raises(ValueError):
            pack_filter_word(*triple)
    for value in (-1, 2**46, True, np.bool_(False), 1.0):
        with pytest.raises(ValueError):
            unpack_filter_word(value)


def test_typed_contracts_and_worker_failures_do_not_report_success():
    for args in [
        (0, "ready", 1),
        (True, "ready", 1),
        (2000001, "ready", 1),
        (1, "CRC-selected", 1),
        (1, "ready", None),
        (1, "no_candidate", 1),
        (1, "ready", True),
        (1, "ready", 8192),
    ]:
        with pytest.raises(ValueError):
            FineService(*args)
    for args in [
        (True, 0, (2,)),
        (256, 2, (2,)),
        (256, 0, [2]),
        (256, 0, (1, 3)),
        (256, 0, (640,)),
        (256, 0, (2,), 5121),
    ]:
        with pytest.raises(ValueError):
            CaptureRequest(*args)
    with pytest.raises(ValueError):
        CaptureTransportAssumptions(window=632, history=384)
    with pytest.raises(ValueError):
        CaptureTransport().estimate(measured_worker=1)
    with pytest.raises(ValueError):
        CaptureTransport().simulate(CausalSource(256), [], fine_worker=7)
    for callback in (lambda e, w: None, lambda e, w: FineService(1, "ready", 2048)):
        with pytest.raises(ValueError):
            CaptureTransport().simulate(
                CausalSource(256), [CaptureRequest(256, 0, (2,))], fine_worker=callback
            )
    for request in (CaptureRequest(256, 0, (300,)), CaptureRequest(256, 0, (2,), 2049)):
        with pytest.raises(ValueError):
            CaptureTransport().simulate(CausalSource(256), [request])


def test_worker_no_candidate_future_wait_and_failures_are_not_crc_answers(monkeypatch):
    profile = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
    template = SyncSignTemplate.from_symbols(profile.sync_symbols())
    worker = CaptureFineWorker(template, 256, confirmation_required=False)

    def event(peaks):
        return dict(
            window_samples=256,
            after_samples=256,
            path=0,
            geometry=CaptureRequest(256, 0, peaks, 8, 128),
        )

    assert worker(event(()), [0] * 256).disposition == "no_candidate"
    assert worker(event((2,)), [0] * 256).disposition == "below_threshold"
    assert worker(event((131,)), [0] * 256).disposition == "needs_future"
    with pytest.raises(ValueError):
        CaptureFineWorker(template, 376, confirmation_required=False)
    with pytest.raises(ValueError):
        CaptureFineWorker(template, 256, confirmation_required=False, sink=True)
    with pytest.raises(ValueError):
        worker({}, [0] * 256)
    monkeypatch.setattr(
        worker.engine, "run", lambda *a, **k: dict(best=None, cycles=7, numerical_rejection=True)
    )
    assert worker(event((2,)), [0] * 256).disposition == "numerical_reject"
    monkeypatch.setattr(
        worker.engine, "run", lambda *a, **k: dict(best=None, cycles=7, resource_drop=True)
    )
    with pytest.raises(RuntimeError, match="resource abort"):
        worker(event((2,)), [0] * 256)


def test_coherent_mode_keeps_low_differential_scores_pending_not_rejected():
    template = SyncSignTemplate.from_symbols(
        PSKReceiveProfile.ordinary(TxConfig(symbol_rate_mhz=4)).sync_symbols()
    )
    seen = []
    worker = CaptureFineWorker(
        template, 256, confirmation_required=True, sink=lambda e, r: seen.append(r)
    )
    request = CaptureRequest(256, 0, (2,), 8, 128)
    service = worker(
        dict(window_samples=256, after_samples=256, path=0, geometry=request), [0] * 256
    )
    assert not seen[0]["best"]["threshold_pass"]
    assert service.disposition == "confirm_pending"
    assert not worker.estimate()["coherent_confirmation_scheduled"]
    with pytest.raises(ValueError):
        CaptureFineWorker(template, 256, confirmation_required=1)
    events = []
    r = CaptureTransport(CaptureTransportAssumptions(clock_hz=100000000, history=256)).simulate(
        CausalSource(2048),
        [request],
        fine_worker=lambda e, w: FineService(5000, "confirm_pending", 8),
        sink=lambda e, w: events.append(e),
    )
    assert r["worker_results"]["confirm_pending_retention_lost"] == 1
    assert r["worker_results"]["ready"] == r["worker_results"]["below_threshold"] == 0
    assert events[0]["continuation_gap_samples"] > 0
