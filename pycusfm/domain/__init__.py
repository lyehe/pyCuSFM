# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""
Domain Layer - Enterprise Business Rules

This layer contains the core business entities, value objects, and domain logic.
It has NO dependencies on other layers - pure Python only.
"""

from .entities.sfm_project import SfMProject
from .entities.camera_rig import CameraRig, Camera
from .entities.frame_metadata import FrameMetadata
from .entities.feature_collection import FeatureCollection
from .entities.pose_graph import PoseGraph
from .entities.keypoint_map import KeypointMap

from .value_objects.pose import Pose
from .value_objects.camera_intrinsics import CameraIntrinsics
from .value_objects.feature_descriptor import FeatureDescriptor
from .value_objects.match_pair import MatchPair

from .enums.feature_type import FeatureType
from .enums.pipeline_mode import PipelineMode
from .enums.coordinate_frame import CoordinateFrame

from .exceptions.domain_exceptions import DomainException
from .exceptions.validation_errors import ValidationError

__all__ = [
    # Entities
    'SfMProject',
    'CameraRig',
    'Camera',
    'FrameMetadata',
    'FeatureCollection',
    'PoseGraph',
    'KeypointMap',
    # Value Objects
    'Pose',
    'CameraIntrinsics',
    'FeatureDescriptor',
    'MatchPair',
    # Enums
    'FeatureType',
    'PipelineMode',
    'CoordinateFrame',
    # Exceptions
    'DomainException',
    'ValidationError',
]
