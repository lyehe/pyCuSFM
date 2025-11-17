# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Feature Collection Entity"""

from dataclasses import dataclass, field
from typing import List, Dict
from ..value_objects.feature_descriptor import FeatureDescriptor
from ..enums.feature_type import FeatureType
from ..exceptions.domain_exceptions import InsufficientFeaturesException


@dataclass
class FeatureCollection:
    """
    Feature collection entity.

    Represents a collection of features extracted from a single image frame.

    Attributes:
        frame_id: Identifier of the frame these features belong to
        feature_type: Type of features (SIFT, ALIKED, etc.)
        features: List of feature descriptors
        metadata: Additional metadata (e.g., extraction parameters)
    """

    frame_id: str
    feature_type: FeatureType
    features: List[FeatureDescriptor] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate feature collection after initialization."""
        if not self.frame_id:
            raise ValueError("Frame ID cannot be empty")

    def add_feature(self, feature: FeatureDescriptor) -> None:
        """
        Add a feature to the collection.

        Args:
            feature: Feature descriptor to add
        """
        self.features.append(feature)

    def get_feature(self, keypoint_id: int) -> FeatureDescriptor:
        """
        Get a feature by its keypoint ID.

        Args:
            keypoint_id: Keypoint identifier

        Returns:
            FeatureDescriptor: The feature

        Raises:
            KeyError: If feature not found
        """
        for feature in self.features:
            if feature.keypoint_id == keypoint_id:
                return feature
        raise KeyError(f"Feature with keypoint_id {keypoint_id} not found")

    @property
    def num_features(self) -> int:
        """Get number of features in collection."""
        return len(self.features)

    @property
    def is_empty(self) -> bool:
        """Check if collection is empty."""
        return len(self.features) == 0

    def validate_minimum_features(self, min_features: int = 8) -> None:
        """
        Validate that there are enough features.

        Args:
            min_features: Minimum required features

        Raises:
            InsufficientFeaturesException: If not enough features
        """
        if self.num_features < min_features:
            raise InsufficientFeaturesException(
                f"Frame {self.frame_id} has only {self.num_features} features, "
                f"minimum {min_features} required",
                details={
                    'frame_id': self.frame_id,
                    'num_features': self.num_features,
                    'min_features': min_features
                })

    def get_top_features(self, n: int) -> List[FeatureDescriptor]:
        """
        Get top N features by response strength.

        Args:
            n: Number of features to return

        Returns:
            List of top N features sorted by response (descending)
        """
        sorted_features = sorted(self.features,
                                 key=lambda f: f.response,
                                 reverse=True)
        return sorted_features[:n]

    def __repr__(self) -> str:
        """String representation."""
        return (f"FeatureCollection(frame={self.frame_id}, "
                f"type={self.feature_type.value}, "
                f"count={self.num_features})")
