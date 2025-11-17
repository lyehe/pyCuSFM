# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Frame Metadata Entity"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from ..value_objects.pose import Pose


@dataclass
class FrameMetadata:
    """
    Frame metadata entity.

    Represents metadata for a single image frame including
    its identifier, file path, timestamp, and optional pose.

    Attributes:
        frame_id: Unique identifier for this frame
        camera_id: Identifier of the camera that captured this frame
        image_path: Path to the image file
        timestamp_us: Capture timestamp in microseconds
        pose: Optional camera pose (from SLAM or ground truth)
        track_id: Optional track identifier (for multi-track scenarios)
        is_keyframe: Whether this frame is selected as a keyframe
    """

    frame_id: str
    camera_id: str
    image_path: Path
    timestamp_us: int
    pose: Optional[Pose] = None
    track_id: Optional[str] = None
    is_keyframe: bool = False

    def __post_init__(self):
        """Validate frame metadata after initialization."""
        if not self.frame_id:
            raise ValueError("Frame ID cannot be empty")
        if not self.camera_id:
            raise ValueError("Camera ID cannot be empty")
        if self.timestamp_us < 0:
            raise ValueError(
                f"Timestamp must be non-negative, got {self.timestamp_us}")

    def has_pose(self) -> bool:
        """Check if frame has an associated pose."""
        return self.pose is not None

    def mark_as_keyframe(self) -> None:
        """Mark this frame as a keyframe."""
        self.is_keyframe = True

    def unmark_keyframe(self) -> None:
        """Unmark this frame as a keyframe."""
        self.is_keyframe = False

    def update_pose(self, pose: Pose) -> None:
        """
        Update the pose for this frame.

        Args:
            pose: New pose to set

        Raises:
            ValueError: If pose timestamp doesn't match frame timestamp
        """
        if pose.timestamp_us != self.timestamp_us:
            raise ValueError(
                f"Pose timestamp ({pose.timestamp_us}) must match "
                f"frame timestamp ({self.timestamp_us})")
        self.pose = pose

    @property
    def image_exists(self) -> bool:
        """Check if the image file exists."""
        return self.image_path.exists()

    def __repr__(self) -> str:
        """String representation."""
        pose_str = f", pose={self.pose}" if self.pose else ""
        keyframe_str = " [KEYFRAME]" if self.is_keyframe else ""
        return (f"FrameMetadata(id={self.frame_id}, camera={self.camera_id}, "
                f"t={self.timestamp_us}{pose_str}{keyframe_str})")
