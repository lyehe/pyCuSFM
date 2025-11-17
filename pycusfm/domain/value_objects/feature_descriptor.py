# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Feature Descriptor Value Object"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class FeatureDescriptor:
    """
    Feature descriptor value object.

    Represents a single keypoint with its descriptor vector.
    This is an immutable value object.

    Attributes:
        keypoint_id: Unique identifier for this keypoint
        position: (x, y) pixel coordinates
        scale: Scale/size of the keypoint
        orientation: Orientation angle in radians
        descriptor: Feature descriptor vector (128D for SIFT, 256D for ALIKED, etc.)
        response: Detector response (strength/confidence)
    """

    keypoint_id: int
    position: Tuple[float, float]
    scale: float
    orientation: float
    descriptor: np.ndarray
    response: float = 0.0

    def __post_init__(self):
        """Validate feature descriptor after initialization."""
        if len(self.position) != 2:
            raise ValueError("Position must be a 2-tuple (x, y)")
        if self.scale <= 0:
            raise ValueError(f"Scale must be positive, got {self.scale}")
        if not isinstance(self.descriptor, np.ndarray):
            raise ValueError("Descriptor must be a numpy array")
        if len(self.descriptor.shape) != 1:
            raise ValueError("Descriptor must be a 1D array")

    @property
    def x(self) -> float:
        """Get x coordinate."""
        return self.position[0]

    @property
    def y(self) -> float:
        """Get y coordinate."""
        return self.position[1]

    @property
    def descriptor_dimension(self) -> int:
        """Get descriptor dimensionality."""
        return self.descriptor.shape[0]

    def distance_to(self, other: 'FeatureDescriptor',
                    metric: str = 'L2') -> float:
        """
        Compute distance to another feature descriptor.

        Args:
            other: Another feature descriptor
            metric: Distance metric ('L2', 'L1', or 'cosine')

        Returns:
            float: Distance value

        Raises:
            ValueError: If descriptors have different dimensions or unknown metric
        """
        if self.descriptor_dimension != other.descriptor_dimension:
            raise ValueError(
                f"Descriptor dimensions must match: "
                f"{self.descriptor_dimension} != {other.descriptor_dimension}"
            )

        if metric == 'L2':
            return float(np.linalg.norm(self.descriptor - other.descriptor))
        elif metric == 'L1':
            return float(np.sum(np.abs(self.descriptor - other.descriptor)))
        elif metric == 'cosine':
            dot = np.dot(self.descriptor, other.descriptor)
            norm1 = np.linalg.norm(self.descriptor)
            norm2 = np.linalg.norm(other.descriptor)
            return float(1.0 - dot / (norm1 * norm2))
        else:
            raise ValueError(
                f"Unknown metric: {metric}. Use 'L2', 'L1', or 'cosine'")

    def __repr__(self) -> str:
        """String representation."""
        return (f"FeatureDescriptor(id={self.keypoint_id}, "
                f"pos=({self.x:.1f}, {self.y:.1f}), "
                f"scale={self.scale:.2f}, dim={self.descriptor_dimension})")
