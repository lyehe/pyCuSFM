# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Match Pair Value Object"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class MatchPair:
    """
    Image pair match value object.

    Represents a pair of images that should be matched for feature correspondences.
    This is an immutable value object.

    Attributes:
        image1_id: Identifier for first image
        image2_id: Identifier for second image
        score: Matching score/confidence (higher is better)
        is_stereo_pair: Whether this is a stereo pair
        overlap_estimate: Estimated visual overlap (0.0 to 1.0)
    """

    image1_id: str
    image2_id: str
    score: float = 1.0
    is_stereo_pair: bool = False
    overlap_estimate: float = 0.0

    def __post_init__(self):
        """Validate match pair after initialization."""
        if self.image1_id == self.image2_id:
            raise ValueError("Cannot match an image with itself")
        if self.score < 0:
            raise ValueError(f"Score must be non-negative, got {self.score}")
        if not (0.0 <= self.overlap_estimate <= 1.0):
            raise ValueError(
                f"Overlap estimate must be in [0, 1], got {self.overlap_estimate}"
            )

    @property
    def image_ids(self) -> Tuple[str, str]:
        """Get both image IDs as a tuple."""
        return (self.image1_id, self.image2_id)

    @property
    def sorted_image_ids(self) -> Tuple[str, str]:
        """Get image IDs in sorted order (for consistent hashing)."""
        return tuple(sorted([self.image1_id, self.image2_id]))

    def contains_image(self, image_id: str) -> bool:
        """
        Check if this pair contains a specific image.

        Args:
            image_id: Image identifier to check

        Returns:
            bool: True if the image is part of this pair
        """
        return image_id in (self.image1_id, self.image2_id)

    def get_other_image(self, image_id: str) -> str:
        """
        Get the other image in the pair.

        Args:
            image_id: One of the images in the pair

        Returns:
            str: The other image identifier

        Raises:
            ValueError: If the image is not part of this pair
        """
        if image_id == self.image1_id:
            return self.image2_id
        elif image_id == self.image2_id:
            return self.image1_id
        else:
            raise ValueError(f"Image {image_id} is not part of this pair")

    def __hash__(self) -> int:
        """Hash based on sorted image IDs (order-independent)."""
        return hash(self.sorted_image_ids)

    def __eq__(self, other) -> bool:
        """
        Check equality based on sorted image IDs.

        Two pairs are equal if they contain the same images,
        regardless of order.
        """
        if not isinstance(other, MatchPair):
            return False
        return self.sorted_image_ids == other.sorted_image_ids

    def __repr__(self) -> str:
        """String representation."""
        stereo_marker = " [STEREO]" if self.is_stereo_pair else ""
        return (f"MatchPair({self.image1_id} <-> {self.image2_id}, "
                f"score={self.score:.3f}{stereo_marker})")
