"""Real source clock→bounded query context→original window→fine;no host selector map."""

import copy

import pytest

from nearlink_sdr.sim.banked_scaled_core import BankedScaledCore
from nearlink_sdr.sim.capture_fine import CaptureFineWorker
from nearlink_sdr.sim.capture_transport import CaptureTransport, CaptureTransportAssumptions
from nearlink_sdr.sim.coarse_capture_source import CoarseCaptureSource
from nearlink_sdr.sim.spline_score import SyncSignTemplate
from nearlink_sdr.sim.word_arithmetic import WordFabricAssumptions
from tests.coarse_capture_reference import stimulus
from tests.test_coarse_ingress import profile


def run(frame=2, *, clock=200, groups=(), blocked=False, actual_fine=True, defect=None):
    p, packed, releases, flags, producer, _ = stimulus(frame, clock=clock, groups=groups)
    if defect == "invalid":
        cycle, tag, _ = releases[900]
        releases[900] = cycle, tag, None
    elif defect == "gap":
        releases.pop(900)
    elif defect == "flag_tail":
        flags.pop()
    events, fine_windows = [], []
    source = CoarseCaptureSource(p.sync_symbols(), iter(releases), iter(flags),
                                 sample_limit=len(packed), source_horizon=producer["cycles"] + 8192,
                                 lanes=16 if frame == 4 else 8)
    engine = CaptureFineWorker(
        SyncSignTemplate.from_symbols(p.sync_symbols()), source.flow.core.window,
        confirmation_required=True, reuse_symbol_power=True, defer_all_future=True,
        word_fabric=WordFabricAssumptions(2), gauss_difference=True, symbol_pipeline=True,
    )

    def worker(event, words):
        end, window = event["after_samples"], event["window_samples"]
        assert list(words) == packed[end - window:end]
        fine_windows.append((end, tuple(words)))
        return engine(event, words)

    transport = CaptureTransport(CaptureTransportAssumptions(
        paths=1, history=4096, window=source.flow.core.window, clock_hz=clock * 1000000))
    result = transport.simulate(
        None, (), source_owner=source, released_requests=True,
        fine_worker=worker if actual_fine else None,
        stalls=((0, 200000),) if blocked else (), sink=lambda e, _: events.append(e),
    )
    return result, source, events, fine_windows, producer


@pytest.mark.parametrize("frame", [2, 3, 4])
def test_actual_RRC_release_and_original_query_window_reach_real_fine_worker(frame):
    result, source, events, windows, producer = run(frame)
    assert result["source_coarse_transport_clock_composed"]
    assert result["input_samples"] == source.stats["rrc_pulses"] == producer["emitted"] == 1536
    assert source.stats["flag_pulses"] == 1536 and source.flow.core.committed_count == 1536
    assert result["live_coarse_failure"] == result["source_epoch_drops"] == 0
    assert source.done and result["pending_descriptors"] == 0
    assert windows and result["completed"] > 0
    assert result["requests"] == result["completed"] + result["queue_drops"]
    assert result["peak_owned"] <= 3 and result["peak_source_operations_per_primitive"] <= 2
    assert all(e["status"] == "completed" for e in events if e["fine_cycles"] is not None)
    assert not result["input_periodic_surrogate_used"] and not result["whole_system_accepted"]
    assert result["carrier_frame_and_decoder_cycles"] is None
    additional = (source.estimate()["own_register_bits_upper"] + 256
                  + result["source_transport_new_register_bits"])
    assert additional == 1517 <= 2048


def test_fixed_random_host_groups_preserve_RRC_values_clock_geometry_and_fine_results():
    a = run(actual_fine=False)
    b = run(groups=(1, 47, 13, 311, 9, 101, 3), actual_fine=False)
    assert a[0] == b[0] and a[1].stats == b[1].stats and a[2:4] == b[2:4]


@pytest.mark.parametrize("clock", [100, 160])
def test_RRC_overload_counts_all_remaining_pulses_without_renumbering_or_delivery(clock):
    result, source, _, _, producer = run(clock=clock, actual_fine=False)
    assert result["live_coarse_failure"] and not result["live_coarse_lossless"]
    assert result["input_samples"] == source.stats["rrc_pulses"] == producer["emitted"]
    assert source.stats["flag_pulses"] == 1536
    assert source.flow.stats["rrc_missing_tags"] == producer["dropped"]
    assert result["completed"] == 0 and source.done


@pytest.mark.parametrize("defect", ["invalid", "gap", "flag_tail"])
def test_invalid_missing_or_truncated_source_cancels_owners_and_counts_tail(defect):
    result, source, events, _, producer = run(actual_fine=False, defect=defect)
    assert result["live_coarse_failure"] and source.done
    assert source.stats["rrc_pulses"] == producer["emitted"] - int(defect == "gap")
    assert source.stats["flag_pulses"] == 1536 - int(defect == "flag_tail")
    assert source.fault_cycle is not None
    assert all(e["cycle"] < source.fault_cycle for e in events if e["status"] == "completed")
    assert all(e["cycle"] == source.fault_cycle
               for e in events if e["status"] == "source_epoch_aborted")
    if defect == "invalid":
        assert result["source_invalid_pulses"] == 1
    elif defect == "gap":
        assert result["source_gap_samples"] == 1


def test_blocked_sink_keeps_input_running_and_counts_finite_owned_slot_drops():
    result, source, _, _, _ = run(blocked=True, actual_fine=False)
    assert result["input_samples"] == 1536 and source.done
    assert result["peak_owned"] == 3 and result["queue_drops"] > 0
    assert not result["adc_backpressure"]


def test_compiled_endpoint_format_rejects_context_errors_before_consuming_query():
    with pytest.raises(ValueError, match="compiled"):
        BankedScaledCore(profile().sync_symbols(), preserve_source_end=1)
    for mode in (False, True):
        core = BankedScaledCore(profile().sync_symbols(), preserve_source_end=mode)
        old = copy.deepcopy(core.stats)
        with pytest.raises(ValueError, match="endpoint"):
            core.request_query(0, 0, source_end=1)
        assert core.next_query_id == 0 and core.stats == old
    old = BankedScaledCore(profile().sync_symbols()).estimate()
    new = BankedScaledCore(profile().sync_symbols(), preserve_source_end=True).estimate()
    assert new["logical_state_bits_upper"] - old["logical_state_bits_upper"] == 256


@pytest.mark.parametrize("rrc,flags", [
    ([(0, 0, 2**46)], []), ([(0, 0, True)], []),
    ([(0, 0, 0), (0, 1, 0)], []), ([], [(0, 0, 1)]),
])
def test_malformed_source_ports_never_become_successful_capture(rrc, flags):
    with pytest.raises(ValueError):
        source = CoarseCaptureSource(profile().sync_symbols(), iter(rrc), iter(flags),
                                     sample_limit=8, source_horizon=100)
        source.advance(0)


def test_clock_EOF_and_finite_statistics_exhaustion_are_explicit():
    source = CoarseCaptureSource(profile().sync_symbols(), (), (),
                                 sample_limit=8, source_horizon=100)
    with pytest.raises(ValueError, match="clock"):
        source.advance(1)
    for cycle in range(645):
        assert source.advance(cycle) == (None, None)
    assert source.done and source.stats["rrc_pulses"] == source.stats["flag_pulses"] == 0
    assert source.advance(645) == (None, None)
    with pytest.raises(ValueError, match="clock"):
        source.advance(645)
    other = CoarseCaptureSource(profile().sync_symbols(), ((0, 0, 0),), (),
                                sample_limit=8, source_horizon=100)
    other.stats["rrc_pulses"] = 2**64 - 1
    with pytest.raises(OverflowError, match="counter"):
        other.advance(0)


def test_source_mode_refuses_unmatched_static_or_replay_contract():
    with pytest.raises(ValueError, match="bounds"):
        CoarseCaptureSource(profile().sync_symbols(), (), (),
                            sample_limit=0, source_horizon=100)
    source = CoarseCaptureSource(profile().sync_symbols(), (), (),
                                 sample_limit=8, source_horizon=100)
    transport = CaptureTransport(CaptureTransportAssumptions(paths=1, history=4096))
    with pytest.raises(ValueError, match="fresh matched"):
        transport.simulate([[0]], (), source_owner=source, released_requests=True)
    assert source.cycle == 0 and source.stats["rrc_pulses"] == 0
