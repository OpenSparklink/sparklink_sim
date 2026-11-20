"""Explicit power/TED overlap in the existing ordered timing/normalization join.

Head CRC and legal frame geometry stay on clock;payload remains an explicit
off-clock service. No new search,phase grid,loop coefficient or source history.
"""

from nearlink_sdr.sim.data_joint_replay import DataJointFrameReplay
from nearlink_sdr.sim.timing_overlap_flow import PowerTEDPacketTimingService

__all__ = ["PowerTEDFrameReplay"]


class PowerTEDFrameReplay(DataJointFrameReplay):
    """Compiled schedule selection before packet ownership;one data pair credit."""

    timing_service_type = PowerTEDPacketTimingService
