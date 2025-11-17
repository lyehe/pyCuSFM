# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Pose Value Object"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class Pose:
    """
    6DOF pose (position + orientation) value object.

    This is an immutable value object representing a camera or vehicle pose.
    Once created, it cannot be modified. Any transformation creates a new Pose.

    Attributes:
        position: (x, y, z) coordinates in meters
        orientation: Quaternion (w, x, y, z) - normalized
        timestamp_us: Timestamp in microseconds
        frame: Coordinate frame identifier (e.g., "camera_frame", "vehicle_frame")
    """

    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]  # Quaternion (w, x, y, z)
    timestamp_us: int
    frame: str = "camera_frame"

    def __post_init__(self):
        """Validate the pose after initialization."""
        # Validate position
        if len(self.position) != 3:
            raise ValueError("Position must be a 3-tuple (x, y, z)")

        # Validate orientation (quaternion)
        if len(self.orientation) != 4:
            raise ValueError(
                "Orientation must be a 4-tuple quaternion (w, x, y, z)")

        # Check quaternion is normalized (approximately)
        quat_norm = sum(q * q for q in self.orientation)**0.5
        if not (0.99 < quat_norm < 1.01):
            raise ValueError(
                f"Quaternion must be normalized (norm={quat_norm:.4f})")

        # Validate timestamp
        if self.timestamp_us < 0:
            raise ValueError("Timestamp must be non-negative")

    @property
    def x(self) -> float:
        """Get x coordinate."""
        return self.position[0]

    @property
    def y(self) -> float:
        """Get y coordinate."""
        return self.position[1]

    @property
    def z(self) -> float:
        """Get z coordinate."""
        return self.position[2]

    @property
    def qw(self) -> float:
        """Get quaternion w component."""
        return self.orientation[0]

    @property
    def qx(self) -> float:
        """Get quaternion x component."""
        return self.orientation[1]

    @property
    def qy(self) -> float:
        """Get quaternion y component."""
        return self.orientation[2]

    @property
    def qz(self) -> float:
        """Get quaternion z component."""
        return self.orientation[3]

    def to_matrix(self) -> np.ndarray:
        """
        Convert pose to 4x4 transformation matrix.

        Returns:
            np.ndarray: 4x4 homogeneous transformation matrix
        """
        # Quaternion to rotation matrix
        w, x, y, z = self.orientation

        # Rotation matrix from quaternion
        R = np.array([
            [
                1 - 2 * (y**2 + z**2), 2 * (x * y - w * z),
                2 * (x * z + w * y)
            ],
            [
                2 * (x * y + w * z), 1 - 2 * (x**2 + z**2),
                2 * (y * z - w * x)
            ],
            [
                2 * (x * z - w * y), 2 * (y * z + w * x),
                1 - 2 * (x**2 + y**2)
            ],
        ])

        # Build 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = self.position

        return T

    @staticmethod
    def from_matrix(matrix: np.ndarray,
                    timestamp_us: int,
                    frame: str = "camera_frame") -> 'Pose':
        """
        Create Pose from 4x4 transformation matrix.

        Args:
            matrix: 4x4 homogeneous transformation matrix
            timestamp_us: Timestamp in microseconds
            frame: Coordinate frame identifier

        Returns:
            Pose: New pose object

        Raises:
            ValueError: If matrix is not 4x4
        """
        if matrix.shape != (4, 4):
            raise ValueError("Matrix must be 4x4")

        # Extract position
        position = tuple(matrix[:3, 3])

        # Extract rotation matrix and convert to quaternion
        R = matrix[:3, :3]
        orientation = Pose._rotation_matrix_to_quaternion(R)

        return Pose(position=position,
                    orientation=orientation,
                    timestamp_us=timestamp_us,
                    frame=frame)

    @staticmethod
    def _rotation_matrix_to_quaternion(
            R: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Convert rotation matrix to quaternion.

        Args:
            R: 3x3 rotation matrix

        Returns:
            Tuple[float, float, float, float]: Quaternion (w, x, y, z)
        """
        # Shepperd's method for numerical stability
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        # Normalize
        norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
        return (w / norm, x / norm, y / norm, z / norm)

    def transform(self, other: 'Pose') -> 'Pose':
        """
        Apply this pose transformation to another pose.

        Args:
            other: Pose to transform

        Returns:
            Pose: Transformed pose (self * other)
        """
        T1 = self.to_matrix()
        T2 = other.to_matrix()
        T_result = T1 @ T2

        return Pose.from_matrix(matrix=T_result,
                                timestamp_us=other.timestamp_us,
                                frame=self.frame)

    def inverse(self) -> 'Pose':
        """
        Compute the inverse of this pose.

        Returns:
            Pose: Inverted pose
        """
        T = self.to_matrix()
        T_inv = np.linalg.inv(T)

        return Pose.from_matrix(matrix=T_inv,
                                timestamp_us=self.timestamp_us,
                                frame=self.frame)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Pose(pos=({self.x:.3f}, {self.y:.3f}, {self.z:.3f}), "
            f"quat=({self.qw:.3f}, {self.qx:.3f}, {self.qy:.3f}, {self.qz:.3f}), "
            f"t={self.timestamp_us}, frame={self.frame})")
