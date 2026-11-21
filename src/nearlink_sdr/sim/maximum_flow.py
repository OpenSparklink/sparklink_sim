"""Registered exponent-histogram services; no score/RRC/whole-RX cycle credit.

Two compiled counter stores share the same functional owner. The four-stage
register layout needs2R2W count ports. The seven-stage serial layout performs
two registered reads and two serialized writes on a1R1W counter RAM.
"""

from nearlink_sdr.phy.acquisition_maximum import ExponentWindow

__all__ = ["ExponentMaximumService"]


class ExponentMaximumService:
    """One descriptor and one output credit,finite clock/counters,explicit abort.

    ``issue`` is a ready/valid handshake;False means the event was NOT consumed.
    An unpausable producer must retain it in a separately charged FIFO or abort
    the epoch. It cannot skip that event and continue this histogram.
    """

    _reasons = (
        None,
        "explicit-abort",
        "invalid-descriptor",
        "counter-overflow",
        "upstream-resource",
    )

    def __init__(self, terms, *, counter_store="register2r2w", start_cycle=0):
        if counter_store not in ("register2r2w", "serial1r1w"):
            raise ValueError("Need a compiled register2r2w or serial1r1w store")
        if type(start_cycle) is not int or not 0 <= start_cycle < 2**63 - 8:
            raise ValueError("Need unsigned63 start clock with pipeline headroom")
        self.window = ExponentWindow(terms)
        self._owner = object()
        self.window._owner = self._owner
        self.counter_store, self.cycle = counter_store, start_cycle
        self.mode, self.stage, self._reason_code = "initializing", 0, 0
        self.request = self.read = self.prepared = self.output = None
        self._registered_count = None
        self._retired_cycle = -1
        self.last_ports = (0, 0)
        self._clear_position, self.last_reset_words = 0, 0
        self.stats = dict(accepted=0, completed=0, held_edges=0, rejected=0)

    @property
    def ready(self):
        return self.mode == "ready"

    @property
    def reason(self):
        """Host decoding of a fixed3-bit code;arbitrary error strings are not stored."""
        return self._reasons[self._reason_code]

    @property
    def latency(self):
        return 4 if self.counter_store == "register2r2w" else 7

    def _increment(self, name):
        value = self.stats[name] + 1
        if not 0 <= value < 2**64:
            raise OverflowError("Unsigned64 service statistics exhausted")
        self.stats[name] = value

    def issue(self, tag, outgoing, incoming):
        self._check_owner()
        if self.mode in ("failed", "finished"):
            raise ValueError("Need a new explicit histogram epoch")
        if self.mode != "idle" or self.cycle == self._retired_cycle:
            self._increment("rejected")
            return False
        if self.cycle >= 2**63 - self.latency - 1:
            raise OverflowError("Insufficient unsigned63 pipeline clock headroom")
        request = self.window.request(tag, outgoing, incoming)
        self._increment("accepted")
        self.request, self.mode, self.stage = request, "working", 0
        return True

    def _check_owner(self):
        if self.window._owner is not self._owner:
            raise ValueError("Histogram ownership changed during clocked service")
        if self.mode == "working":
            committed = self.stage >= (3 if self.latency == 4 else 6)
            if self.window.next_tag != self.request[0] + int(committed):
                raise ValueError("Histogram tag changed outside the owning pipeline")
        elif self.ready and self.window.next_tag != self.output[0] + 1:
            raise ValueError("Completed histogram output lost its owning context")

    def _read_one(self, pair):
        phase = self.request[1]
        self.last_ports = self.last_ports[0] + 1, self.last_ports[1]
        return self.window.counts[phase][pair[0]]

    def _write_one(self, index):
        changes = self.prepared[1]
        if index < len(changes):
            key, value = changes[index]
            self.window.counts[self.request[1]][key] = value
            self.last_ports = self.last_ports[0], self.last_ports[1] + 1

    def step(self):
        if self.cycle >= 2**63 - 1:
            raise OverflowError("Unsigned63 service clock exhausted")
        self._check_owner()
        edge = self.cycle + 1
        self.last_ports = (0, 0)
        self.last_reset_words = 0
        try:
            if self.mode == "initializing":
                if self.latency == 7 and self._clear_position < 644:
                    phase, key = divmod(self._clear_position, 161)
                    self.window.counts[phase][key] = 0
                    self._clear_position += 1
                    self.last_ports = (0, 1)
                else:
                    if self.latency == 4:
                        for phase in range(4):
                            self.window.counts[phase][:] = [0] * 161
                        self.last_reset_words = 644  # Decoded one-bit reset pulse.
                    self.window.bitmaps[:] = self.window.groups[:] = [0] * 4
                    self.window.totals[:] = [0] * 4
                    self.mode = "idle"
            elif self.mode == "working":
                stage = self.stage + 1
                if self.latency == 4:
                    if stage == 1:
                        self.read = self.window.read(self.request)
                        self.last_ports = (2, 0)
                    elif stage == 2:
                        self.prepared = self.window.prepare(self.read)
                    elif stage == 3:
                        self.window.commit(self.prepared, owner=self._owner)
                        self.last_ports = (0, len(self.prepared[1]))
                else:
                    # The first count is not visible on the read-issue edge.
                    if stage == 1:
                        self._registered_count = self._read_one(self.request[2])
                    elif stage == 2:
                        self.read = (self.request, self._registered_count)
                        self._registered_count = self._read_one(self.request[3])
                    elif stage == 3:
                        phase = self.request[1]
                        self.read = (
                            *self.read,
                            self._registered_count,
                            self.window.bitmaps[phase],
                            self.window.groups[phase],
                            self.window.totals[phase],
                        )
                        self.prepared = self.window.prepare(self.read)
                    elif stage in (4, 5):
                        self._write_one(stage - 4)
                    elif stage == 6:
                        self.window.commit(self.prepared, owner=self._owner, write_counts=False)
                if stage == self.latency:
                    self._increment("completed")
                    self.output = self.request[0], self.window.maximum(self.request[1]), edge
                    self.mode = "ready"
                self.stage = stage
            elif self.ready:
                self._increment("held_edges")
        except (ValueError, OverflowError) as error:
            self.abort(
                "counter-overflow" if isinstance(error, OverflowError) else "invalid-descriptor"
            )
            raise
        self.cycle = edge

    def take(self):
        self._check_owner()
        if not self.ready:
            return None
        result = self.output
        self.request = self.read = self.prepared = self.output = None
        self._registered_count = None
        self.mode, self.stage = "idle", 0
        # A synchronous downstream retirement is not an additional input port:
        # issue is legal on the next edge,as enforced by last-retirement clock.
        self._retired_cycle = self.cycle
        return result

    def finish(self):
        if self.mode != "idle":
            raise ValueError("Drain or explicitly abort the unfinished descriptor before EOF")
        self.mode = "finished"

    def abort(self, reason):
        if not isinstance(reason, str) or not reason:
            raise ValueError("Need an explicit resource or upstream failure reason")
        self._reason_code = self._reasons.index(reason) if reason in self._reasons else 1
        self.mode = "failed"
        self.request = self.read = self.prepared = self.output = None
        self._registered_count = None
        # No zero-cycle clearing of644 counters is claimed. Partial serial
        # writes invalidate this entire epoch;only a fresh owner can restart.

    def estimate(self):
        result = self.window.estimate()
        bits = self.window.counter_bits
        request = 63 + 2 + 2 * 9
        read = request + 2 * bits + 161 + 11 + bits
        prepared = request + 2 * (8 + bits) + 2 + 161 + 11 + bits + 6 * 64
        pipeline = request + read + prepared + 134 + 63 + 6 + 4 * 64 + 64 + 4 + 3
        pipeline += (bits if self.latency == 7 else 0) + 11
        total = result["functional_state_bits"] + pipeline
        result.update(
            counter_store=self.counter_store,
            pipeline_bits_upper=pipeline,
            total_state_bits_upper=total,
            latency_edges=self.latency,
            initiation_interval=self.latency + 1,
            initialization_edges=(1 if self.latency == 4 else 645),
            parallel_reset_word_fanout=(644 if self.latency == 4 else 0),
            reset_counter_address_bits=10,
            bounded_reason_bits=3,
            output_tag_exponent_clock_bits=134,
            per_clock_adders_upper=dict(clock63=1, held64=1, rejected64=1, stage3=1),
            per_event_add64_upper=8,
            initialization_add10_per_clear_edge=int(self.latency == 7),
            counter_address_constant=161,
            counter_address_add10_per_event_upper=8,
            address_packing="phase*161+bin;constant shifts/adds;linear reset RAM address",
            initialization_divmod="host list decoding only;physical reset is a linear address",
            count_ports_per_clock=(
                dict(read=2, write=2) if self.latency == 4 else dict(read=1, write=1)
            ),
            count_RAM18_estimate=int(self.latency == 7),
            count_BRAM36_site_reservation=int(self.latency == 7),
            counter_RAM18_allocated_bits=(18432 if self.latency == 7 else 0),
            counter_RAM18_unused_bits=(
                18432 - result["count_state_bits"] if self.latency == 7 else 0
            ),
            uncredited_sibling_RAM18_bits=(18432 if self.latency == 7 else 0),
            counter_address_width=10,
            address_generators=(2 if self.latency == 4 else 1),
            serial_counter_read_mux2_fabric_upper=(
                0 if self.latency == 7 else result["dynamic_counter_read_mux2_upper"]
            ),
            counter_register_bits=(result["count_state_bits"] if self.latency == 4 else 0),
            other_register_bits_upper=(
                total if self.latency == 4 else total - result["count_state_bits"]
            ),
            DSP=0,
            new_IQ_FIFO_bits=0,
            research_cap_bits=8192,
            research_cap_scope="logical live state;RAM padding/site and wholeRX billed separately",
            research_cap_pass=total <= 8192,
            LUT=None,
            FF_placement=None,
            Fmax=None,
            count_RAM_packing_scope="one1024x18 SDP half;other half not credited or shared",
            full_scaled_bank_cycles=None,
            full_RX_cycles=None,
            hardware_assumption="one-cycle registered counts,bitmap update and priority stage",
        )
        return result
