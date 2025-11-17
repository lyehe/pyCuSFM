# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Feature Type Enumeration"""

from enum import Enum


class FeatureType(Enum):
    """
    Supported feature detector and descriptor types.

    Each feature type has different characteristics:
    - SIFT: Traditional, CPU-based, scale-invariant
    - ALIKED: Deep learning, TensorRT-accelerated
    - SUPERPOINT: Deep learning, TensorRT-accelerated
    """

    SIFT = "sift"
    ALIKED = "aliked"
    SUPERPOINT = "superpoint"

    @property
    def requires_tensorrt(self) -> bool:
        """
        Check if this feature type requires TensorRT runtime.

        Returns:
            bool: True if TensorRT is needed, False otherwise
        """
        return self in (FeatureType.ALIKED, FeatureType.SUPERPOINT)

    @property
    def model_directory_name(self) -> str:
        """
        Get the model directory name for this feature type.

        Returns:
            str: Directory name under models/ or empty string if not applicable

        Examples:
            >>> FeatureType.ALIKED.model_directory_name
            'aliked_lightglue'
            >>> FeatureType.SIFT.model_directory_name
            ''
        """
        if self == FeatureType.ALIKED:
            return "aliked_lightglue"
        elif self == FeatureType.SUPERPOINT:
            return "superpoint_lightglue"
        return ""

    @property
    def uses_lightglue_matcher(self) -> bool:
        """
        Check if this feature type uses LightGlue for matching.

        Returns:
            bool: True if LightGlue matcher is used
        """
        return self in (FeatureType.ALIKED, FeatureType.SUPERPOINT)

    def __str__(self) -> str:
        """String representation."""
        return self.value
