"""One chronological actual selection and replayed source,never picked by CRC."""

from functools import partial
from unittest.mock import patch

from nearlink_sdr.sim.capture_fine import pack_filter_word
from nearlink_sdr.sim.coherent_interleaved import ScheduledCoherentCapturedFrameReplay
from nearlink_sdr.sim.fine_selection import FineSelection
from tests.captured_frame_reference import receive
from tests.test_coherent_captured_frame import profile_case


def captured(ft=2, **options):
    profile, iq, quiet, expected = profile_case(ft, **options)
    words = []

    def observe(*values):
        value = pack_filter_word(*values)
        words.append(value)
        return value

    factory = partial(ScheduledCoherentCapturedFrameReplay,
                      structure="interleaved-signed22x18")
    with patch("tests.captured_frame_reference.pack_filter_word", observe), patch(
        "tests.captured_frame_reference.CapturedFrameReplay", factory
    ):
        _, frames, _, _, _, _ = receive(profile, iq, quiet, retire_before_fine=True)
    first = frames[0][0]  # Unconditional first chronological terminal,including any CRC failure.
    selected = FineSelection(**first["upstream_selection"])
    edge = first["coherent_word_service"]["issued_cycle"]
    return profile, words, expected, selected, edge


def drive(owner, selection, edge, words, *, blocked=(), raw_end=None, after_tick=None):
    """Pausable fixed source for component checks,not a continuous ADC experiment."""
    owner.begin(selection, edge)
    end = len(words) if raw_end is None else raw_end
    while not owner.done:
        cycle = owner.cycle + 1
        stopped = any(a <= cycle < b for a, b in blocked)
        owner.tick(cycle, blocked=stopped)
        if after_tick is not None:
            after_tick(owner)
        if owner.needs_word and not stopped:
            index = owner.source_index
            if index < end:
                owner.issue_read(index, words[index], cycle)
            else:
                owner.end_input(cycle)
        assert owner.cycle < edge + 1_000_000
    return owner.take()


def existing_at_copy_edges(profile, selection, reads, last_cycle, *, receipt=None):
    """Public existing owner independently consumes the recorded actual copy port.

    This is a structural/controller oracle,not an independent Gardner arithmetic
    or RF oracle. No receiver gets expected bytes or a CRC-dependent selection.
    """
    from nearlink_sdr.sim.initialized_frame_replay import InitializedFrameReplay
    from nearlink_sdr.sim.receive_history import ReceiveHistoryPort

    origin = selection.source_end - selection.window
    first = selection.start_tick // 8 - 1
    future = (selection.start_tick // 8 + 4 * (len(selection.words) - 1) + 3
              + bool(selection.start_tick % 8))
    history = ReceiveHistoryPort(1024, first_index=first)
    by_edge = {r["returned_cycle"]: r for r in reads}
    assert len(by_edge) == len(reads)
    fields = selection.reference_selection()
    owner = None
    for cycle in range(reads[0]["returned_cycle"], last_cycle + 1):
        row = by_edge.get(cycle)
        if row is not None:
            assert history.end == row["index"] - origin
            if owner is None:
                assert history.write(history.end, row["word"], cycle)
            else:
                assert owner.cycle == cycle and owner.write_input(row["word"])
        if owner is None and history.end == future:
            owner = InitializedFrameReplay(
                profile, history, start_cycle=cycle, **fields,
                sample_limit=2**20, coherent_threshold=None,
                confirmed_receipt=receipt,
            )
        if owner is not None:
            assert owner.cycle == cycle
            if owner.pending_service is not None:
                owner.resolve_functional()
            if owner.done:
                assert cycle == last_cycle
                break
            owner.step()
    assert owner is not None and owner.done
    return owner


def observed_receipt(profile, selection, evidence):
    """Reconstruct saved actual finite receipt for diagnostics,never rescore/select."""
    from nearlink_sdr.phy.sync_fit_word import compile_sync_phases
    from nearlink_sdr.sim.coherent_word_flow import CoherentWordReceipt

    n, d = evidence["numerator"], evidence["denominator"]
    return CoherentWordReceipt(selection,
        compile_sync_phases(profile.sync_symbols()),
        evidence["angle_turn40"], evidence["step_word20"],
        tuple(evidence["word_correlation"]), evidence["signal_energy"], n, d,
        (20*n > 9*d)-(20*n < 9*d), evidence["ready_cycle"])


def same_age_confirmed_float(profile, words, selection, receipt):
    """Same-age float arithmetic uses the same actual gate,not a repeated85% gate."""
    from nearlink_sdr.phy.selected_rx import SelectedPSKReceiver
    from nearlink_sdr.sim.initialized_frame_replay import InitializedFrameReplay
    from tests.test_feedback_timing_replay import run

    def confirm(owner, sync, delta, score, threshold):
        row = InitializedFrameReplay.checked_word_confirmation(receipt, owner.profile,
            start_cycle=receipt.available_cycle, start_tick=selection["start_tick"],
            delta=delta, differential_score=score, selected_sync=sync, threshold=threshold)
        return row, receipt.accepted

    with patch.object(SelectedPSKReceiver, "_confirm_selected_context", confirm):
        return run(profile, words, selection, age=1, finite=False)
