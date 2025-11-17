# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Camera Rig Entity"""

from dataclasses import dataclass, field
from typing import List, Optional
from ..value_objects.camera_intrinsics import CameraIntrinsics
from ..value_objects.pose import Pose
from ..exceptions.domain_exceptions import InvalidCameraConfigurationException


@dataclass
class Camera:
    """
    Individual camera entity.

    Represents a single camera with its intrinsic parameters and
    optional extrinsic parameters (relative pose in rig).

    Attributes:
        camera_id: Unique identifier for this camera
        intrinsics: Camera calibration parameters
        extrinsics: Optional pose relative to rig reference frame
    """

    camera_id: str
    intrinsics: CameraIntrinsics
    extrinsics: Optional[Pose] = None

    def __post_init__(self):
        """Validate camera after initialization."""
        if not self.camera_id:
            raise ValueError("Camera ID cannot be empty")

    def has_extrinsics(self) -> bool:
        """Check if camera has extrinsic calibration."""
        return self.extrinsics is not None

    def __repr__(self) -> str:
        """String representation."""
        extrinsics_str = f", extrinsics={self.extrinsics}" if self.extrinsics else ""
        return f"Camera(id={self.camera_id}, {self.intrinsics}{extrinsics_str})"


@dataclass
class CameraRig:
    """
    Camera rig entity.

    Represents a single camera or a set of synchronized cameras (e.g., stereo pair).

    Business rules:
    - Stereo pairs must have exactly 2 cameras
    - Stereo pairs must have positive baseline
    - All cameras in rig share synchronized timestamps

    Attributes:
        rig_id: Unique identifier for this rig
        cameras: List of cameras in this rig
        is_stereo: Whether this is a stereo camera rig
        baseline: Stereo baseline in meters (required for stereo rigs)
    """

    rig_id: str
    cameras: List[Camera] = field(default_factory=list)
    is_stereo: bool = False
    baseline: Optional[float] = None

    def __post_init__(self):
        """Validate camera rig after initialization."""
        if not self.rig_id:
            raise ValueError("Rig ID cannot be empty")

    def add_camera(self, camera: Camera) -> None:
        """
        Add a camera to this rig.

        Args:
            camera: Camera to add

        Raises:
            ValueError: If camera ID already exists in rig
        """
        if self.has_camera(camera.camera_id):
            raise ValueError(
                f"Camera {camera.camera_id} already exists in rig {self.rig_id}"
            )
        self.cameras.append(camera)

    def get_camera(self, camera_id: str) -> Optional[Camera]:
        """
        Get a camera by ID.

        Args:
            camera_id: Camera identifier

        Returns:
            Camera or None if not found
        """
        for camera in self.cameras:
            if camera.camera_id == camera_id:
                return camera
        return None

    def has_camera(self, camera_id: str) -> bool:
        """
        Check if rig contains a specific camera.

        Args:
            camera_id: Camera identifier

        Returns:
            bool: True if camera exists in rig
        """
        return self.get_camera(camera_id) is not None

    @property
    def num_cameras(self) -> int:
        """Get number of cameras in rig."""
        return len(self.cameras)

    def validate_stereo_configuration(self) -> None:
        """
        Validate stereo camera configuration.

        Raises:
            InvalidCameraConfigurationException: If stereo configuration is invalid
        """
        if not self.is_stereo:
            return

        errors = []

        # Must have exactly 2 cameras
        if len(self.cameras) != 2:
            errors.append(
                f"Stereo rig must have exactly 2 cameras, got {len(self.cameras)}"
            )

        # Must have positive baseline
        if self.baseline is None or self.baseline <= 0:
            errors.append(
                f"Stereo rig must have positive baseline, got {self.baseline}"
            )

        # Both cameras should have extrinsics
        cameras_without_extrinsics = [
            c.camera_id for c in self.cameras if not c.has_extrinsics()
        ]
        if cameras_without_extrinsics:
            errors.append(
                f"Stereo cameras missing extrinsics: {cameras_without_extrinsics}"
            )

        if errors:
            raise InvalidCameraConfigurationException(
                f"Invalid stereo configuration for rig {self.rig_id}",
                details={'errors': errors})

    def validate(self) -> None:
        """
        Validate camera rig configuration.

        Raises:
            InvalidCameraConfigurationException: If configuration is invalid
        """
        if not self.cameras:
            raise InvalidCameraConfigurationException(
                f"Camera rig {self.rig_id} must have at least one camera")

        # Validate stereo configuration if applicable
        if self.is_stereo:
            self.validate_stereo_configuration()

        # Check for duplicate camera IDs
        camera_ids = [c.camera_id for c in self.cameras]
        if len(camera_ids) != len(set(camera_ids)):
            duplicates = [
                cid for cid in camera_ids if camera_ids.count(cid) > 1
            ]
            raise InvalidCameraConfigurationException(
                f"Duplicate camera IDs in rig {self.rig_id}: {duplicates}")

    def __repr__(self) -> str:
        """String representation."""
        stereo_str = f", stereo baseline={self.baseline}m" if self.is_stereo else ""
        return f"CameraRig(id={self.rig_id}, cameras={self.num_cameras}{stereo_str})"
