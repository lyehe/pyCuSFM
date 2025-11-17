# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""SfM Project Entity - Aggregate Root"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path
from ..enums.pipeline_mode import PipelineMode
from ..enums.feature_type import FeatureType
from .camera_rig import CameraRig
from .frame_metadata import FrameMetadata
from .feature_collection import FeatureCollection
from .pose_graph import PoseGraph
from .keypoint_map import KeypointMap
from ..exceptions.validation_errors import EntityValidationError


@dataclass
class SfMProject:
    """
    Structure-from-Motion Project - Aggregate Root.

    This is the main aggregate root that represents an entire SfM reconstruction project.
    It coordinates all entities and enforces business invariants.

    Attributes:
        project_id: Unique identifier for this project
        input_directory: Directory containing input images
        workspace_directory: Directory for output and intermediate files
        mode: Pipeline execution mode (Isaac, AV, Localization, Multi-track)
        feature_type: Type of features to extract (SIFT, ALIKED, SuperPoint)
        camera_rigs: List of camera rigs in this project
        frames: List of frame metadata
        feature_collections: Dictionary mapping frame_id to FeatureCollection
        pose_graph: Optional pose graph for optimization
        keypoint_map: Optional 3D reconstruction map
        config_params: Additional configuration parameters
    """

    project_id: str
    input_directory: Path
    workspace_directory: Path
    mode: PipelineMode
    feature_type: FeatureType = FeatureType.SIFT
    camera_rigs: List[CameraRig] = field(default_factory=list)
    frames: List[FrameMetadata] = field(default_factory=list)
    feature_collections: Dict[str, FeatureCollection] = field(
        default_factory=dict)
    pose_graph: Optional[PoseGraph] = None
    keypoint_map: Optional[KeypointMap] = None
    config_params: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate project after initialization."""
        if not self.project_id:
            raise ValueError("Project ID cannot be empty")

    # Camera Rig Management
    def add_camera_rig(self, rig: CameraRig) -> None:
        """
        Add a camera rig to the project.

        Args:
            rig: Camera rig to add

        Raises:
            ValueError: If rig ID already exists
        """
        if self.has_rig(rig.rig_id):
            raise ValueError(
                f"Camera rig {rig.rig_id} already exists in project")
        rig.validate()
        self.camera_rigs.append(rig)

    def get_rig(self, rig_id: str) -> Optional[CameraRig]:
        """Get a camera rig by ID."""
        for rig in self.camera_rigs:
            if rig.rig_id == rig_id:
                return rig
        return None

    def has_rig(self, rig_id: str) -> bool:
        """Check if project has a specific rig."""
        return self.get_rig(rig_id) is not None

    @property
    def num_rigs(self) -> int:
        """Get number of camera rigs."""
        return len(self.camera_rigs)

    # Frame Management
    def add_frame(self, frame: FrameMetadata) -> None:
        """
        Add a frame to the project.

        Args:
            frame: Frame metadata to add

        Raises:
            ValueError: If frame ID already exists
        """
        if self.has_frame(frame.frame_id):
            raise ValueError(
                f"Frame {frame.frame_id} already exists in project")
        self.frames.append(frame)

    def get_frame(self, frame_id: str) -> Optional[FrameMetadata]:
        """Get a frame by ID."""
        for frame in self.frames:
            if frame.frame_id == frame_id:
                return frame
        return None

    def has_frame(self, frame_id: str) -> bool:
        """Check if project has a specific frame."""
        return self.get_frame(frame_id) is not None

    def get_frames_by_camera(self, camera_id: str) -> List[FrameMetadata]:
        """Get all frames from a specific camera."""
        return [
            frame for frame in self.frames if frame.camera_id == camera_id
        ]

    def get_keyframes(self) -> List[FrameMetadata]:
        """Get all frames marked as keyframes."""
        return [frame for frame in self.frames if frame.is_keyframe]

    @property
    def num_frames(self) -> int:
        """Get total number of frames."""
        return len(self.frames)

    @property
    def num_keyframes(self) -> int:
        """Get number of keyframes."""
        return len(self.get_keyframes())

    # Feature Management
    def add_feature_collection(self,
                                collection: FeatureCollection) -> None:
        """
        Add a feature collection for a frame.

        Args:
            collection: Feature collection to add

        Raises:
            ValueError: If frame doesn't exist or collection already exists
        """
        if not self.has_frame(collection.frame_id):
            raise ValueError(
                f"Frame {collection.frame_id} does not exist in project")
        if collection.frame_id in self.feature_collections:
            raise ValueError(
                f"Feature collection for frame {collection.frame_id} already exists"
            )
        self.feature_collections[collection.frame_id] = collection

    def get_feature_collection(
            self, frame_id: str) -> Optional[FeatureCollection]:
        """Get feature collection for a frame."""
        return self.feature_collections.get(frame_id)

    def has_features(self, frame_id: str) -> bool:
        """Check if frame has extracted features."""
        return frame_id in self.feature_collections

    @property
    def num_frames_with_features(self) -> int:
        """Get number of frames with extracted features."""
        return len(self.feature_collections)

    # Pose Graph Management
    def initialize_pose_graph(self) -> None:
        """Initialize the pose graph for this project."""
        self.pose_graph = PoseGraph(project_id=self.project_id)

    def has_pose_graph(self) -> bool:
        """Check if project has a pose graph."""
        return self.pose_graph is not None

    # Keypoint Map Management
    def initialize_keypoint_map(self) -> None:
        """Initialize the 3D keypoint map for this project."""
        self.keypoint_map = KeypointMap(project_id=self.project_id)

    def has_keypoint_map(self) -> bool:
        """Check if project has a keypoint map."""
        return self.keypoint_map is not None

    # Validation
    def validate(self) -> None:
        """
        Validate project consistency.

        Raises:
            EntityValidationError: If project is invalid
        """
        errors = []

        # Check input directory exists
        if not self.input_directory.exists():
            errors.append(
                f"Input directory does not exist: {self.input_directory}")

        # Check workspace directory exists (or can be created)
        if not self.workspace_directory.exists():
            try:
                self.workspace_directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(
                    f"Cannot create workspace directory: {self.workspace_directory} - {e}"
                )

        # Must have at least one camera rig
        if not self.camera_rigs:
            errors.append("Project must have at least one camera rig")

        # Validate each camera rig
        for rig in self.camera_rigs:
            try:
                rig.validate()
            except Exception as e:
                errors.append(f"Invalid camera rig {rig.rig_id}: {e}")

        # Check that all frames reference valid cameras
        camera_ids = set()
        for rig in self.camera_rigs:
            for camera in rig.cameras:
                camera_ids.add(camera.camera_id)

        for frame in self.frames:
            if frame.camera_id not in camera_ids:
                errors.append(
                    f"Frame {frame.frame_id} references unknown camera {frame.camera_id}"
                )

        # Check that all feature collections reference valid frames
        frame_ids = {frame.frame_id for frame in self.frames}
        for collection_frame_id in self.feature_collections.keys():
            if collection_frame_id not in frame_ids:
                errors.append(
                    f"Feature collection references unknown frame {collection_frame_id}"
                )

        if errors:
            raise EntityValidationError(entity_type="SfMProject",
                                        entity_id=self.project_id,
                                        errors=errors)

    # Statistics
    def get_statistics(self) -> Dict:
        """
        Get project statistics.

        Returns:
            Dictionary with project statistics
        """
        stats = {
            'project_id': self.project_id,
            'mode': self.mode.value,
            'feature_type': self.feature_type.value,
            'num_rigs': self.num_rigs,
            'num_frames': self.num_frames,
            'num_keyframes': self.num_keyframes,
            'num_frames_with_features': self.num_frames_with_features,
            'has_pose_graph': self.has_pose_graph(),
            'has_keypoint_map': self.has_keypoint_map(),
        }

        if self.pose_graph:
            stats['pose_graph_nodes'] = self.pose_graph.num_nodes
            stats['pose_graph_edges'] = self.pose_graph.num_edges

        if self.keypoint_map:
            stats['map_points'] = self.keypoint_map.num_points
            stats['map_observations'] = self.keypoint_map.get_total_observations(
            )

        return stats

    def __repr__(self) -> str:
        """String representation."""
        return (f"SfMProject(id={self.project_id}, mode={self.mode.value}, "
                f"rigs={self.num_rigs}, frames={self.num_frames}, "
                f"keyframes={self.num_keyframes})")
