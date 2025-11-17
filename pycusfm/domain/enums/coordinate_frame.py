# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Coordinate Frame Enumeration"""

from enum import Enum


class CoordinateFrame(Enum):
    """
    Coordinate frame types for pose representation.

    Different coordinate frames are used depending on the context:
    - CAMERA_FRAME: Individual camera coordinate system
    - VEHICLE_FRAME: Vehicle/robot body coordinate system
    - VEHICLE_RIG: Multi-camera rig coordinate system
    - WORLD_FRAME: Global/world coordinate system
    """

    CAMERA_FRAME = "camera_frame"
    VEHICLE_FRAME = "vehicle_frame"
    VEHICLE_RIG = "vehicle_rig"
    WORLD_FRAME = "world_frame"

    @property
    def supports_extrinsics_optimization(self) -> bool:
        """
        Check if this frame type supports extrinsics optimization.

        Returns:
            bool: True if extrinsics can be optimized in this frame
        """
        return self == CoordinateFrame.VEHICLE_RIG

    @property
    def is_local_frame(self) -> bool:
        """
        Check if this is a local (relative) coordinate frame.

        Returns:
            bool: True if local frame, False if global
        """
        return self in (
            CoordinateFrame.CAMERA_FRAME,
            CoordinateFrame.VEHICLE_FRAME,
            CoordinateFrame.VEHICLE_RIG,
        )

    @property
    def requires_transformation(self) -> bool:
        """
        Check if transformations between frames are required.

        Returns:
            bool: True if frame transformations are needed
        """
        return self != CoordinateFrame.CAMERA_FRAME

    def __str__(self) -> str:
        """String representation."""
        return self.value
