# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Pipeline Mode Enumeration"""

from enum import Enum


class PipelineMode(Enum):
    """
    Pipeline execution modes for different use cases.

    Each mode has different characteristics and requirements:
    - ISAAC_ROBOTICS: Stereo cameras with synchronized frames (Isaac platform)
    - AUTONOMOUS_VEHICLE: Rolling video with flexible frame selection (AV)
    - LOCALIZATION: Match against pre-built map for relocalization
    - MULTI_TRACK: Multiple camera rigs or time-separated acquisitions
    """

    ISAAC_ROBOTICS = "isaac"
    AUTONOMOUS_VEHICLE = "av"
    LOCALIZATION = "localization"
    MULTI_TRACK = "multi_track"

    @property
    def config_subdirectory(self) -> str:
        """
        Get the configuration subdirectory for this mode.

        Returns:
            str: Subdirectory name under configs/

        Examples:
            >>> PipelineMode.ISAAC_ROBOTICS.config_subdirectory
            'isaac'
            >>> PipelineMode.AUTONOMOUS_VEHICLE.config_subdirectory
            'av'
        """
        if self == PipelineMode.ISAAC_ROBOTICS:
            return "isaac"
        elif self == PipelineMode.AUTONOMOUS_VEHICLE:
            return "av"
        # For LOCALIZATION and MULTI_TRACK, use isaac configs by default
        return "isaac"

    @property
    def requires_stereo_calibration(self) -> bool:
        """
        Check if this mode requires stereo camera calibration.

        Returns:
            bool: True if stereo calibration is needed
        """
        return self == PipelineMode.ISAAC_ROBOTICS

    @property
    def supports_frame_sampling(self) -> bool:
        """
        Check if this mode supports dynamic frame sampling.

        Returns:
            bool: True if frame sampling is supported
        """
        # AV mode can skip frame selection
        return self != PipelineMode.AUTONOMOUS_VEHICLE

    @property
    def requires_global_localization(self) -> bool:
        """
        Check if this mode requires global localization.

        Returns:
            bool: True if global localization is needed
        """
        return self in (PipelineMode.LOCALIZATION, PipelineMode.MULTI_TRACK)

    def __str__(self) -> str:
        """String representation."""
        return self.value
