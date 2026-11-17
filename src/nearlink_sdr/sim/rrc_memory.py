"""Concrete finite RRC memory images/ports, not a new filter (§6.2.1.2).

The caller supplies actual input/product words; no expected data participates
in addressing or admission. Replica images, padding and one-cycle product read
registers are explicit. Raw outputs feed the existing alignment stage. Conflicts
produce missing responses and diagnostics, never a hidden extra read port.
"""

from collections import Counter, defaultdict

__all__ = ["RRCMemory"]


class RRCMemory:
    """8source copies or8bank dual/quad LUTRAM,4owned product LUTRAM banks.

    dual: RAM32X1D shared SPO/write address, one independent DPO.
    quad: RAM32M tied two-bit data on all four write slices, independent A/B
    reads; unused C/D outputs still charged. Products use RAM64X1D DPO only.
    Primitive replication is represented by separate arrays, not ideal RAM.
    """

    def __init__(self, raw_mode="quad", buffers=3, *, diagnostic_sink=None):
        if raw_mode not in ("copies", "dual", "quad") or type(buffers) is not int or buffers not in (2, 3, 4):
            raise ValueError("Need explicit raw mode and2..4owned product slots")
        self.raw_mode, self.buffers, self.sink = raw_mode, buffers, diagnostic_sink
        copies = 2 if raw_mode == "dual" else 4
        self.raw = ([[0]*128 for _ in range(8)] if raw_mode == "copies"
                    else [[[0]*32 for _ in range(copies)] for _ in range(8)])
        self.products = [[[0]*64 for _ in range(2)] for _ in range(4)]
        self.pending = []
        self.count, self.cycle = 0, -1
        self.stats = dict(raw_reads=0, startup_zeros=0, input_bypasses=0, expired_reads=0,
                          raw_conflict_banks=0, unserved_raw_reads=0, product_reads=0,
                          product_writes=0, product_hazards=0, source_writes=0,
                          raw_replica_writes=0, product_replica_writes=0,
                          peak_raw_reads_per_bank=0, peak_product_read_registers=0)

    def estimate(self):
        """UG474 uniform mappings, not synthesized LUT/FF/timing results."""
        mode = self.raw_mode
        return dict(raw_mode=mode, raw_bram36=8 if mode == "copies" else 0,
            raw_memory_luts=0 if mode == "copies" else 8*32*2,
            product_memory_luts=4*69*2, product_bram36=0,
            raw_logical_bits=128*32, product_logical_padded_bits=self.buffers*44*69,
            raw_replicated_image_bits=128*32*8 if mode == "copies" else
                8*32*32*(2 if mode == "dual" else 4),
            raw_allocated_primitive_bits=8*36864 if mode == "copies" else 512*64,
            product_allocated_primitive_bits=4*69*64*2,
            raw_ideal_slicem_sites=0 if mode == "copies" else 128,
            product_ideal_slicem_sites=4*35,
            extra_control_and_monitoring_ff=1280,
            product_read_pipeline_bits=4*69, product_read_pipeline_already_in_rrc=True,
            additional_read_pipeline_ff=0, extra_dsp=0,
            raw_routing_mux_bit_inputs_upper=8*16*32,
            raw_current_word_bypass_mux_bit_inputs=8*2*32,
            address_control_lut_count=None, placement_and_timing_verified=False,
            monitoring_updates_scheduled=False, whole_rx_accepted=False)

    @staticmethod
    def _word(value, width):
        if type(value) is not int or not 0 <= value < 2**width:
            raise ValueError("Word outside explicit unsigned port width")
        return value

    def _address(self, request):
        slot, j = request
        if (type(slot) is not int or type(j) is not int or not 0 <= slot < self.buffers
                or not 0 <= j < 41):
            raise ValueError("Unallocated product slot/tap")
        return j % 4, slot*11+j//4

    def _diagnose(self, reason, **fields):
        if self.sink is not None:
            self.sink(dict(cycle=self.cycle, reason=reason, **fields))

    def tick(self, cycle, *, source_write=None, raw_reads=(), product_writes=(), product_reads=()):
        """One clock; source_write=(absolute_index,u32), writes=(slot,j,u69).

        Returns current raw operands and previous product reads (slot,j,word).
        Responses from conflicting banks/addresses are None. The surrounding
        RRC schedule must reject/drop them, not treat them as received data.
        """
        if type(cycle) is not int or cycle != self.cycle+1 or cycle >= 2**60:
            raise ValueError("Need successive clocks inside the finite counter horizon")
        if len(raw_reads) > 8 or len(product_writes) > 4 or len(product_reads) > 4:
            raise ValueError("Access vector exceeds the existing RRC lane bounds")
        self.cycle = cycle
        previous, self.pending = self.pending, []
        write_index = None
        if source_write is not None:
            write_index, word = source_write
            if type(write_index) is not int or write_index != self.count:
                raise ValueError("All source samples must advance in order, including dropped jobs")
            word = self._word(word, 32)
            if self.raw_mode == "copies":
                for replica in self.raw:
                    replica[write_index % 128] = word
                replication = 8
            else:
                for replica in self.raw[write_index % 8]:
                    replica[(write_index//8) % 16] = word
                replication = len(self.raw[0])
            self.count += 1
            self.stats["source_writes"] += 1
            self.stats["raw_replica_writes"] += replication
        result, banks = [None]*len(raw_reads), defaultdict(list)
        for lane, index in enumerate(raw_reads):
            if type(index) is not int or not -80 <= index < self.count:
                raise ValueError("Need an arrived raw sample or defined startup zero")
            self.stats["raw_reads"] += 1
            if index < 0:
                result[lane] = 0
                self.stats["startup_zeros"] += 1
            elif index == write_index:
                result[lane] = word
                self.stats["input_bypasses"] += 1
            elif index < self.count-128:
                self.stats["expired_reads"] += 1
                self._diagnose("raw_expired", operand=index)
            else:
                banks[index % 8].append((lane, index))
        for bank, reads in banks.items():
            self.stats["peak_raw_reads_per_bank"] = max(
                self.stats["peak_raw_reads_per_bank"], len(reads))
            limit = (8 if self.raw_mode == "copies" else
                     1 if self.raw_mode == "dual" and write_index is not None
                     and write_index % 8 == bank else 2)
            if len(reads) > limit:
                self.stats["raw_conflict_banks"] += 1
                self.stats["unserved_raw_reads"] += len(reads)
                self._diagnose("raw_ports", bank=bank, operands=[v for _, v in reads],
                               writer=write_index, available_reads=limit)
                continue
            for port, (lane, index) in enumerate(reads):
                physical_port = (1 if self.raw_mode == "dual" and write_index is not None
                                 and write_index % 8 == bank else port)
                result[lane] = (self.raw[lane][index % 128] if self.raw_mode == "copies"
                                else self.raw[bank][physical_port][(index//8) % 16])
        writes = [(self._address((slot, j)), self._word(value, 69))
                  for slot, j, value in product_writes]
        reads = [(self._address(req), req) for req in product_reads]
        if (max(Counter(bank for (bank, _), _ in writes).values(), default=0) > 1
                or max(Counter(bank for (bank, _), _ in reads).values(), default=0) > 1):
            raise ValueError("At most one independent product write/read per physical bank")
        write_addresses = {address for address, _ in writes}
        # Keep async read stable through its following capture edge, too.
        hazards = write_addresses & ({address for address, _ in reads}
            | {self._address((slot, j)) for slot, j, _ in previous})
        if hazards:
            self.stats["product_hazards"] += len(hazards)
            self._diagnose("product_read_write_overlap", addresses=sorted(hazards))
        previous = [(slot, j, None if self._address((slot, j)) in hazards else value)
                    for slot, j, value in previous]
        for (bank, address), value in writes:
            for replica in self.products[bank]:
                replica[address] = value
            self.stats["product_writes"] += 1
            self.stats["product_replica_writes"] += 2
        for (bank, address), (slot, j) in reads:
            value = None if (bank, address) in hazards else self.products[bank][1][address]
            self.pending.append((slot, j, value))
            self.stats["product_reads"] += 1
        self.stats["peak_product_read_registers"] = max(
            self.stats["peak_product_read_registers"], len(self.pending))
        return dict(raw=result, products=previous)
