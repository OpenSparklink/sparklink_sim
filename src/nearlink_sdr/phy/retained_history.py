"""Finite filtered-sample history and a counted host-only contiguous-view adapter."""

import numpy as np

__all__ = ["RetainedSampleHistory"]


class RetainedSampleHistory:
    """Ring with absolute indices;mirroring serves the existing native float API.

    The logical live history has `capacity` samples. The host allocates TWO
    copies and writes both,so a wrapped live interval is one contiguous view.
    This layout does not require a future hardware ring to duplicate RAM.
    Retirement is supplied by actual interpolation dependencies,not CRC/data.
    """

    def __init__(self, capacity, first_index):
        if (
            type(capacity) is not int
            or not 4 <= capacity <= 4096
            or capacity & (capacity - 1)
            or type(first_index) is not int
            or not 0 <= first_index < 2**20
        ):
            raise ValueError("Need power2 capacity4..4096 and finite first index")
        self.capacity, self.first_index = capacity, first_index
        self.storage = np.empty(2 * capacity, complex)
        self.origin = self.end = first_index
        self.next_input = 0
        self.peak = self.writes = self.retired = self.leading_skipped = 0

    @property
    def count(self):
        return self.end - self.origin

    @property
    def view(self):
        slot = self.origin % self.capacity
        return self.storage[slot : slot + self.count]

    def append(self, index, value):
        """Accept next finite sample,orFalse without overwriting live history."""
        if type(index) is not int or index != self.next_input or index >= 2**20:
            raise ValueError("Need the next bounded absolute sample index")
        if not np.isscalar(value) or not np.isfinite(value):
            raise ValueError("Need one finite sample")
        if index < self.first_index:
            self.next_input += 1
            self.leading_skipped += 1
            return True
        if self.count == self.capacity:
            return False
        if index != self.end:
            raise RuntimeError("Non-contiguous retained history")
        slot = index % self.capacity
        self.storage[slot] = self.storage[slot + self.capacity] = value
        self.end += 1
        self.next_input += 1
        self.writes += 1
        self.peak = max(self.peak, self.count)
        return True

    def retire_before(self, index):
        """Retire observed data only;never create a gap in future arrivals."""
        if type(index) is not int or not 0 <= index <= 2**20:
            raise ValueError("Need a bounded exclusive retirement index")
        target = max(self.origin, min(index, self.end))
        self.retired += target - self.origin
        self.origin = target

    def diagnostics(self):
        return dict(
            capacity_samples=self.capacity,
            origin=self.origin,
            end=self.end,
            live_samples=self.count,
            peak_samples=self.peak,
            leading_skipped=self.leading_skipped,
            retired_samples=self.retired,
            logical_writes=self.writes,
            host_mirror_writes=2 * self.writes,
            host_array_bytes=self.storage.nbytes,
            hardware_port_or_cycle_verified=False,
        )
