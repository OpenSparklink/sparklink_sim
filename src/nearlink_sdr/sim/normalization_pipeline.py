"""Four finite normalization contexts, one4DSP bank and one2R coefficient ROM.

Only scheduling changes relative to SymbolNormalizationService. The same actual
micro-operations carry every value; prediction and owner commit use separate
registered edges. This is not a cache/timing join or whole-RX throughput proof.
"""

from nearlink_sdr.phy.interpolation_word import unpack_interpolated_word
from nearlink_sdr.phy.normalization_word import MAX_SYMBOLS, _integer
from nearlink_sdr.phy.sync_fit_word import INPUT_FORMAT, unsigned
from nearlink_sdr.sim.normalization_word_flow import SymbolNormalizationService

__all__ = ["PipelinedNormalizationService"]

_DSP_ISSUE = frozenset(("interpolate_issue", "gain_issue", "rotate_issue"))


class PipelinedNormalizationService(SymbolNormalizationService):
    """Four register slots; oldest-first arbitration and one ordered output register.

    Each graph stage has its own combinational macro, used at most once/edge.
    All three multiply stages arbitrate the same physical four-lane bank.
    No resource sharing with other RX/TX nodes is credited.
    """

    def __init__(self, state):
        super().__init__(state)
        self.slots = [None] * 4
        self.head = self.tail = self.owned = 0
        self.private_phase, self.private_index = state.phase, state.count
        self.pipeline_stats = dict.fromkeys(
            ["predictions", "admission_backpressure", "DSP_issue_stalls",
             "ROM_issue_stalls", "macro_stalls", "aborted_contexts"], 0
        )
        self.peak_owned = 0
        self._DSP_used = False
        self.serial_mode = False

    def _pipeline_count(self, key, amount=1):
        self.pipeline_stats[key] = unsigned(self.pipeline_stats[key] + amount, 64)

    @property
    def mode(self):
        if self.serial_mode:
            return super().mode
        if self.closed:
            return self.closed
        if self.registered is not None:
            return "ready"
        if self.owned:
            return "pipeline"
        return "idle" if self.state.inverse is not None else "prepare_gain"

    @property
    def can_issue(self):
        if self.serial_mode:
            return super().can_issue
        return (not self.closed and self.owned < 4
                and self.stats["accepted"] < MAX_SYMBOLS
                and self.last_issue_cycle != self.cycle
                and self.last_take_cycle != self.cycle)

    def issue(self, word):
        if self.serial_mode:
            return super().issue(word)
        if self.closed:
            raise RuntimeError("Closed normalization packet cannot resume")
        word = unsigned(_integer(word), 52)
        r, i, e = unpack_interpolated_word(word, INPUT_FORMAT)
        if not self.can_issue:
            self._pipeline_count("admission_backpressure")
            return False
        accepted = unsigned(self.stats["accepted"] + 1, 64)
        if self.slots[self.tail] is not None:
            raise RuntimeError("Normalization tail overwrote an owned slot")
        self.slots[self.tail] = dict(word=word, real=r, imag=i, exponent=e,
                                     issued=self.cycle, stage="predict")
        self.tail = (self.tail + 1) & 3
        self.owned += 1
        self.peak_owned = max(self.peak_owned, self.owned)
        self.stats["accepted"] = accepted
        self.last_issue_cycle = self.cycle
        return True

    def _issue_products(self, pairs, following):
        if self._DSP_used:
            raise RuntimeError("Two normalization stages issued the same DSP bank")
        super()._issue_products(pairs, following)
        self._DSP_used = True

    def step(self):
        if self.serial_mode:
            self._DSP_used = False
            return super().step()
        next_cycle = unsigned(self.cycle + 1, 64)
        if not self.closed:
            inverse = self.state.inverse  # Registered reciprocal, never same-edge forwarding.
            self._prepare_gain(next_cycle)
            self._DSP_used = False
            rom_reads = 0
            stages = set()
            for offset in range(self.owned):
                self.job = self.slots[(self.head + offset) & 3]
                stage = self.job["stage"]
                if stage in stages and stage != "DSP_wait":
                    self._pipeline_count("macro_stalls")
                    continue
                if stage == "register_commit" and (offset or self.registered is not None):
                    continue
                if stage in _DSP_ISSUE:
                    if stage == "gain_issue" and inverse is None:
                        continue
                    if self._DSP_used:
                        self._pipeline_count("DSP_issue_stalls")
                        continue
                reads = 2 if stage == "ROM_first_request" or (
                    stage == "ROM_first_return" and self.state.fmt.mode == "linear"
                ) else 0
                if (stage in ("ROM_first_return", "ROM_second_return")
                        and self.job["ROM_due"] != self.cycle):
                    raise RuntimeError("Registered ROM return lost its one-edge destination")
                if reads and rom_reads:
                    self._pipeline_count("ROM_issue_stalls")
                    continue
                if stage == "predict":
                    self.job.update(index=self.private_index, phase=self.private_phase,
                                    stage="phase")
                    self.private_phase = (
                        self.private_phase + self.state.frequency
                    ) & ((1 << 40) - 1)
                    self.private_index = unsigned(self.private_index + 1, 18)
                    self._pipeline_count("predictions")
                else:
                    self._symbol(inverse, next_cycle)
                if reads:
                    self.job["ROM_due"] = next_cycle
                elif stage in ("ROM_first_return", "ROM_second_return"):
                    self.job.pop("ROM_due")
                rom_reads += reads
                stages.add(stage)
            self.job = self.slots[self.head] if self.owned else None
        self.cycle = next_cycle

    def validate_take(self):
        """Read-only preflight for an eventual joint timing/normalization transfer."""
        if self.registered is None:
            return False
        j = self.job if self.serial_mode else self.slots[self.head]
        if (j is None or self.registered["index"] != self.state.count
                or self.registered["phase_before"] != self.state.phase
                or self.registered["raw_word"] != j["word"]):
            raise RuntimeError("Stale or reordered normalization owner/result")
        unsigned(self.stats["completed"] + 1, 64)
        return True

    def take(self):
        if self.serial_mode:
            return super().take()
        if not self.validate_take():
            return None
        self.job = self.slots[self.head]
        result = super().take()
        self.slots[self.head] = None
        self.head = (self.head + 1) & 3
        self.owned -= 1
        self.job = self.slots[self.head] if self.owned else None
        return result

    def enter_serial(self):
        """Drain before one-way mode transfer;same owner,registers,DSP and ROM bank."""
        if (self.closed or self.serial_mode or self.owned or self.registered is not None
                or self.state.inverse is None or self.private_index != self.state.count):
            raise RuntimeError("Normalization contexts must drain before serial data transfer")
        self.serial_mode = True

    def abort(self):
        if not self.closed:
            self._pipeline_count("aborted_contexts", self.owned
                                 + int(self.serial_mode and self.job is not None))
            super().abort()
            self.slots = [None] * 4
            self.owned = 0
            self.head = self.tail = 0

    @staticmethod
    def estimate(mode="linear"):
        base = SymbolNormalizationService.estimate(mode)
        # Conservative full register allocation, no lifetime overlay is credited.
        context = dict(source_and_tag=52 + 44 + 8 + 18 + 40 + 64 + 5,
                       ROM_work=12 + 16 + 2 * 25 + 68 + 72 + 22 + 64,
                       DSP_external=4 * 43 + 4 * 43 + 64 + 4,
                       coefficient_work=96 + 36 + 11,
                       output_work=82 + 55 + 2)
        parts = {name: base["state_parts"][name] for name in (
            "packet_owner", "registered_commit", "restoring_divider",
            "service_statistics", "clocks_and_held_tags", "control",
        )}
        parts.update(four_contexts=4 * sum(context.values()), predictor=40 + 18,
                     ring_and_peak_occupancy=2 + 2 + 3 + 3,
                     pipeline_statistics=6 * 64, shared_bank_control=1, serial_data_mode=1)
        return base | dict(
            state_parts=parts, context_parts=context,
            state_register_bits_upper=sum(parts.values()), symbol_credits=4,
            input_handoff_edges=1, prepared_symbol_latency_edges=None,
            no_conflict_input_interval_candidate=7 if mode == "linear" else None,
            private_predictor_update_edges=1, output_slots=1,
            per_context_registers_no_unproved_lifetime_reuse=True,
            shared_DSP_bank_issues_per_edge_max=1, ROM_reads_per_edge_max=2,
            stage_macro_instances="one per named graph stage;independent phase,address,"
                "sign,35bit interpolation-RNE,48bit combine,48bit coefficient-normalize,"
                "41bit rotate-sum,41bit output-normalize,owner commit,prediction",
            wide_macro_resources_increased_vs_serial=True,
            context_access="four explicit register slots;oldest-first mux;one input write;"
                "each stage updates a distinct owned slot;not a2port RAM claim",
            actual_cache_broadcast_or_timing_join_included=False,
            after_drain_serial_latency_edges=base["prepared_symbol_latency_edges"],
            after_drain_same_owner_bank_transfer=True,
            synthesized_LUT_FF_or_frequency=None,
        )
