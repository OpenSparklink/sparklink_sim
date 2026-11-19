"""Selected hard header→causal descramble→integer SSC→real CRC→legal geometry.

No demap,capture,carrier,payload or MAC delivery is performed by this module.
All codeword,information,field and context storage is finite and separately charged.
"""

from dataclasses import replace
from itertools import islice

from nearlink_sdr.common.scrambler import _lfsr_step
from nearlink_sdr.phy.control_info import ControlInfoA2, ControlInfoB1, control_field_layout
from nearlink_sdr.phy.control_polar_word import ControlPolarFormat
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile, control_field_limits, control_field_policy
from nearlink_sdr.phy.sync_fit_word import unsigned
from nearlink_sdr.phy.tx_pipeline import head_crc_parameters
from nearlink_sdr.sim.control_polar_flow import ControlPolarService
from nearlink_sdr.sim.header_geometry_flow import HeaderGeometryService

__all__ = ["HeaderWordService"]


class HeaderWordService:
    """One immutable profile,one owned header,one held real-CRC result.

    N-bit hard-demapped input is presented in parallel. It is not an arbitrary IQ
    stream interface. Desc/CRC/field reads are one bit per edge;subservice handoff
    observes the preceding registered ready,never consumes a same-edge new result.
    """

    def __init__(self, profile, *, layout="dense", observer=None):
        if not isinstance(profile, PSKReceiveProfile):
            raise ValueError("Need explicit verified PSK profile context")
        self.profile = replace(profile, config=replace(profile.config))
        self.policy_limits = control_field_limits(self.profile)
        cfg = self.profile.config
        if (
            ((profile.control_format != "A2" or not profile.has_lqi) and cfg.frame_type == 2)
            or profile.mcs_from_lqi
            or profile.control_format not in ("A2", "B1")
        ):
            raise ValueError("Compiled hard header requires ordinary A2+LQI or B1")
        if (
            type(cfg.whitening_seed) is not int
            or not 0 <= cfg.whitening_seed <= 127
            or type(cfg.mcs_index) is not int
            or any(type(mcs) is not int or not 0 <= mcs <= 12 for mcs in self.profile.allowed_mcs)
            or len(self.profile.allowed_mcs) > 13
        ):
            raise ValueError("Need finite whitening,negotiated MCS and permitted MCS set")
        if observer is not None and not callable(observer):
            raise ValueError("Observer must be callable")
        self.observer = observer
        self.poly, self.width, self.seed, mask = head_crc_parameters(cfg)
        self.mask = tuple([0] * self.width if mask is None else map(int, mask))
        self.format = (
            ControlPolarFormat(64, 40) if cfg.frame_type == 2 else ControlPolarFormat(256, 51)
        )
        self.codec = ControlPolarService(self.format, layout=layout, observer=self._codec_event)
        self.geometry = HeaderGeometryService(
            cfg.frame_type, cfg.crc_len, cfg.pilot_interval,
            length_contract=profile.length_contract, observer=self._geometry_event
        )
        self.layout = (
            (("lqi", 8), *control_field_layout("A2"))
            if cfg.frame_type == 2
            else control_field_layout("B1")
        )
        assert sum(width for _, width in self.layout) == cfg.ctrl_bits_len
        self.bits, self.decoded = [0] * self.format.N, (0,) * self.format.K
        self.values, self.result = {}, None
        self.mode = "idle"
        self.index = self.cycle = self.field = self.offset = self.accumulator = 0
        self.whitening_state = self.crc_state = self.mismatch = 0
        self.stats = dict.fromkeys(
            (
                "accepted",
                "completed",
                "aborted",
                "input_stalls",
                "output_stalls",
                "descramble_bits",
                "CRC_input_bits",
                "CRC_parity_bits",
                "field_bits",
                "policy_edges",
            ),
            0,
        )

    def _count(self, key):
        self.stats[key] = unsigned(self.stats[key] + 1, 64)

    def _emit(self, event, **fields):
        if self.observer:
            self.observer(dict(cycle=self.cycle, event=event, **fields))

    def _codec_event(self, event):
        self._emit("SSC", local=event)

    def _geometry_event(self, event):
        self._emit("geometry", local=event)

    @property
    def ready(self):
        return self.mode == "ready"

    @property
    def input_ready(self):
        return self.mode == "idle"

    def issue(self, bits):
        bits = list(islice(bits, self.format.N + 1))
        if len(bits) != self.format.N or any(
            type(bit) is not int or bit not in (0, 1) for bit in bits
        ):
            raise ValueError("Need exactly one finite binary hard header word")
        if self.mode == "aborted":
            raise RuntimeError("Reset aborted header before reuse")
        if not self.input_ready:
            self._count("input_stalls")
            return False
        unsigned(self.stats["completed"] + 1, 64)
        self._count("accepted")
        self.bits[:], self.values, self.result = bits, {}, None
        self.index = self.field = self.offset = self.accumulator = self.mismatch = 0
        self.whitening_state = self.profile.config.whitening_seed
        self.mode = "descramble"
        return True

    def _complete(self, row):
        self._count("completed")
        self.result = row
        self.mode = "ready"

    def _failure(self, status, **extra):
        self._complete(
            dict(
                status=status,
                header_crc=False,
                data_crc=None,
                length_ok=None,
                payload=None,
                **extra,
            )
        )

    def step(self):
        unsigned(self.cycle + 1, 64)
        mode = self.mode
        # Capture only ready values visible BEFORE advancing either subservice.
        codec_result = self.codec.take() if mode == "codec" and self.codec.ready else None
        geometry_result = (
            self.geometry.take() if mode == "geometry" and self.geometry.ready else None
        )
        self.codec.step()
        self.geometry.step()
        if mode == "ready":
            self._count("output_stalls")
        elif mode == "descramble":
            bit, self.whitening_state = _lfsr_step(self.whitening_state)
            self.bits[self.index] ^= bit
            self._emit(
                "descramble",
                index=self.index,
                value=self.bits[self.index],
                state=self.whitening_state,
            )
            self._count("descramble_bits")
            self.index += 1
            if self.index == self.format.N:
                self.mode = "start_codec"
        elif mode == "start_codec":
            if not self.codec.issue(self.bits):
                raise RuntimeError("Header owns exactly one idle codec context")
            self.mode = "codec"
        elif mode == "codec" and codec_result is not None:
            assert codec_result["crc_status"] == "not_checked"
            self.decoded = codec_result["information_bits"]
            self.crc_state, self.index, self.mode = self.seed, 0, "CRC_input"
        elif mode == "CRC_input":
            feedback = self.decoded[self.index] ^ ((self.crc_state >> (self.width - 1)) & 1)
            self.crc_state = ((self.crc_state << 1) & ((1 << self.width) - 1)) ^ (
                self.poly if feedback else 0
            )
            self._emit(
                "CRC_input", index=self.index, value=self.decoded[self.index], state=self.crc_state
            )
            self._count("CRC_input_bits")
            self.index += 1
            if self.index == self.profile.config.ctrl_bits_len:
                self.index, self.mode = 0, "CRC_parity"
        elif mode == "CRC_parity":
            expected = ((self.crc_state >> (self.width - 1 - self.index)) & 1) ^ self.mask[
                self.index
            ]
            received = self.decoded[self.profile.config.ctrl_bits_len + self.index]
            self.mismatch |= expected ^ received
            self._emit("CRC_parity", index=self.index, expected=expected, received=received)
            self._count("CRC_parity_bits")
            self.index += 1
            if self.index == self.width:
                if self.mismatch:
                    self._failure("bad_header_crc")
                else:
                    self.index, self.mode = 0, "fields"
        elif mode == "fields":
            name, width = self.layout[self.field]
            self.accumulator |= self.decoded[self.index] << self.offset
            self._count("field_bits")
            self._emit("field", name=name, bit=self.offset, value=self.decoded[self.index])
            self.index += 1
            self.offset += 1
            if self.offset == width:
                self.values[name] = self.accumulator
                self.field += 1
                self.offset = self.accumulator = 0
                if self.field == len(self.layout):
                    self.mode = "policy"
        elif mode == "policy":
            self._count("policy_edges")
            values = self.values.copy()
            lqi = values.pop("lqi", None)
            head = (ControlInfoA2 if self.profile.control_format == "A2" else ControlInfoB1)(
                **values
            )
            policy = control_field_policy(self.profile, head, lqi, limits=self.policy_limits)
            self.result = dict(
                header_crc=True,
                header=values,
                lqi=lqi,
                mcs=policy["mcs"],
                mcs_source=policy["mcs_source"],
                length_ok=policy["length_ok"],
                header_end_symbol=self.geometry.header_end,
                status=policy["status"],
                data_crc=None,
                payload=None,
            )
            if policy["status"] is not None:
                self._complete(self.result)
            else:
                self.mode = "start_geometry"
        elif mode == "start_geometry":
            if not self.geometry.issue(self.values["data_length"], self.result["mcs"]):
                raise RuntimeError("Header owns exactly one idle geometry context")
            self.mode = "geometry"
        elif mode == "geometry" and geometry_result is not None:
            self.result.update(geometry_result, status="header_valid",
                               length_contract=self.profile.length_contract)
            self._complete(self.result)
        self.cycle += 1

    def take(self):
        if not self.ready:
            return None
        row, self.result, self.mode = self.result, None, "idle"
        return row

    def abort(self):
        if self.mode not in ("idle", "aborted"):
            self._count("aborted")
            self.codec.abort()
            self.geometry.abort()
            self.mode, self.result = "aborted", None

    def reset(self):
        if self.mode not in ("idle", "aborted"):
            raise RuntimeError("Drain or abort before header reset")
        self.codec.reset()
        self.geometry.reset()
        self.mode = "idle"

    def cost(self):
        """Incremental physical/logic accounting,not complete receiver acceptance."""
        from nearlink_sdr.sim.control_polar_flow import control_polar_cost

        codec = control_polar_cost(self.format, layout=self.codec.layout)
        geometry = self.geometry.cost()
        extra = dict(
            input_word=self.format.N,
            decoded_word=self.format.K,
            decoded_fields=self.profile.config.ctrl_bits_len,
            held_geometry_receipt=6 * 18 + self.profile.config.ctrl_bits_len + 16 + 12,
            CRC=self.width + 1,
            scrambler=7,
            indices_and_controller=64,
            counters=len(self.stats) * 64,
            cycle=64,
            profile_context_upper=256,
        )
        logical = codec["logical_total_bits"] + geometry["logical_total_bits"] + sum(extra.values())
        return dict(
            codec=codec,
            geometry=geometry,
            extra_state_bits=extra,
            logical_total_bits=logical,
            DSP=0,
            RAMB18_equivalents=codec["RAMB18_equivalents"],
            BRAM36_sites_upper=codec["BRAM36_sites_upper"],
            CRC_shift_bits=self.width,
            CRC_compare_bits_per_edge=1,
            field_source_read_bits_per_edge=1,
            input_bus_bits=self.format.N,
            bit_read_mux2_upper=(self.format.N - 1 + self.format.K - 1 + 2 * (self.width - 1)),
            input_bit_write_mux2_upper=self.format.N,
            field_descriptor_ROM_bits=10 * 8,
            policy_compare_bits=11,
            policy_parallel_length_comparators=3,
            policy_compiled_minimum_bits=3,
            policy_compiled_maximum_bits=11,
            length_contract_bits=1,
            profile_context_suballocated_bits=237,
            profile_reservation_contains_descriptor_ROM=True,
            profile_context_allocation_bits=dict(
                CRC_polynomial=24, CRC_seed=24, CRC_mask=24, CRC_width=5,
                whitening_seed=7, negotiated_MCS=4, allowed_MCS_mask=13, format=2,
                control_bit_count=6, layout=2, abort_reason=3, field_descriptor_ROM=80,
                max_payload_bytes=11, hasLQI=1, B1_flag=1, data_CRC_width=6,
                pilot_interval=5, frame_type=3, payload_bitorder=1,
                max_wire_length=11, min_nonzero_wire_length=3, length_contract=1,
            ),
            policy_membership_mux_inputs=13,
            policy_membership_selector_bits=4,
            policy_membership_selector_slots=16,
            policy_membership_constant_zero_slots=3,
            CRC_AND_XOR_gates_upper=2 * self.width,
            additional_raw_sample_ports=0,
            demap_cycles=None,
            payload_cycles=None,
            default_changed=False,
            full_RX_acceptance=False,
            synthesis=False,
            capacity_candidate_met=(
                logical <= 98304
                and codec["RAMB18_equivalents"] <= 8
                and codec["BRAM36_sites_upper"] <= 7
            ),
        )
