# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Feature Extraction DTOs"""

from dataclasses import dataclass
from pathlib import Path
from ...domain.entities.sfm_project import SfMProject
from ...domain.enums.feature_type import FeatureType


@dataclass
class FeatureExtractionRequest:
    """
    Input data for feature extraction use case.

    Attributes:
        project: SfM project containing frames to process
        output_directory: Where to write extracted features
        feature_type: Type of features to extract
        max_keypoints: Maximum number of keypoints (-1 for config default)
        batch_size: Batch size for GPU processing (0 for auto)
        enable_debug: Whether to output debug information
        config_path: Path to feature extraction config file
    """

    project: SfMProject
    output_directory: Path
    feature_type: FeatureType = FeatureType.SIFT
    max_keypoints: int = -1
    batch_size: int = 0
    enable_debug: bool = False
    config_path: Path = None


@dataclass
class FeatureExtractionResponse:
    """
    Output data from feature extraction use case.

    Attributes:
        features_directory: Directory containing extracted features
        num_images_processed: Number of images that had features extracted
        total_keypoints: Total number of keypoints extracted
        average_keypoints_per_image: Average keypoints per image
        execution_time_seconds: Time taken for extraction
    """

    features_directory: Path
    num_images_processed: int
    total_keypoints: int = 0
    average_keypoints_per_image: float = 0.0
    execution_time_seconds: float = 0.0
