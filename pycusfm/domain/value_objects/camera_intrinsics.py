# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Camera Intrinsics Value Object"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """
    Camera intrinsic parameters value object.

    This is an immutable value object representing camera calibration parameters.

    Attributes:
        fx: Focal length in x (pixels)
        fy: Focal length in y (pixels)
        cx: Principal point x coordinate (pixels)
        cy: Principal point y coordinate (pixels)
        width: Image width in pixels
        height: Image height in pixels
        distortion_coeffs: Optional distortion coefficients (k1, k2, p1, p2, k3)
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion_coeffs: Optional[Tuple[float, ...]] = None

    def __post_init__(self):
        """Validate camera intrinsics after initialization."""
        if self.fx <= 0:
            raise ValueError(f"fx must be positive, got {self.fx}")
        if self.fy <= 0:
            raise ValueError(f"fy must be positive, got {self.fy}")
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")
        if self.cx < 0 or self.cx >= self.width:
            raise ValueError(
                f"cx must be in [0, {self.width}), got {self.cx}")
        if self.cy < 0 or self.cy >= self.height:
            raise ValueError(
                f"cy must be in [0, {self.height}), got {self.cy}")

    def to_matrix(self) -> np.ndarray:
        """
        Convert to 3x3 camera calibration matrix (K matrix).

        Returns:
            np.ndarray: 3x3 intrinsic calibration matrix
                [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]
        """
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy],
                         [0, 0, 1]])

    @staticmethod
    def from_matrix(matrix: np.ndarray,
                    width: int,
                    height: int,
                    distortion_coeffs: Optional[Tuple[float,
                                                      ...]] = None
                    ) -> 'CameraIntrinsics':
        """
        Create CameraIntrinsics from 3x3 calibration matrix.

        Args:
            matrix: 3x3 camera calibration matrix
            width: Image width in pixels
            height: Image height in pixels
            distortion_coeffs: Optional distortion coefficients

        Returns:
            CameraIntrinsics: New camera intrinsics object

        Raises:
            ValueError: If matrix is not 3x3
        """
        if matrix.shape != (3, 3):
            raise ValueError("Camera matrix must be 3x3")

        fx = matrix[0, 0]
        fy = matrix[1, 1]
        cx = matrix[0, 2]
        cy = matrix[1, 2]

        return CameraIntrinsics(fx=fx,
                                fy=fy,
                                cx=cx,
                                cy=cy,
                                width=width,
                                height=height,
                                distortion_coeffs=distortion_coeffs)

    @property
    def aspect_ratio(self) -> float:
        """
        Get camera aspect ratio.

        Returns:
            float: Aspect ratio (width / height)
        """
        return self.width / self.height

    @property
    def fov_x_degrees(self) -> float:
        """
        Get horizontal field of view in degrees.

        Returns:
            float: Horizontal FOV in degrees
        """
        return 2 * np.arctan(self.width / (2 * self.fx)) * 180 / np.pi

    @property
    def fov_y_degrees(self) -> float:
        """
        Get vertical field of view in degrees.

        Returns:
            float: Vertical FOV in degrees
        """
        return 2 * np.arctan(self.height / (2 * self.fy)) * 180 / np.pi

    def scale(self, factor: float) -> 'CameraIntrinsics':
        """
        Scale camera intrinsics (for image resizing).

        Args:
            factor: Scale factor (e.g., 0.5 for half resolution)

        Returns:
            CameraIntrinsics: Scaled intrinsics
        """
        return CameraIntrinsics(
            fx=self.fx * factor,
            fy=self.fy * factor,
            cx=self.cx * factor,
            cy=self.cy * factor,
            width=int(self.width * factor),
            height=int(self.height * factor),
            distortion_coeffs=self.distortion_coeffs)

    def __repr__(self) -> str:
        """String representation."""
        return (f"CameraIntrinsics(fx={self.fx:.1f}, fy={self.fy:.1f}, "
                f"cx={self.cx:.1f}, cy={self.cy:.1f}, "
                f"size={self.width}x{self.height})")
