# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SfMProject entity"""

import pytest
from pathlib import Path
from pycusfm.domain.entities.sfm_project import SfMProject
from pycusfm.domain.entities.camera_rig import CameraRig, Camera
from pycusfm.domain.entities.frame_metadata import FrameMetadata
from pycusfm.domain.entities.feature_collection import FeatureCollection
from pycusfm.domain.value_objects.camera_intrinsics import CameraIntrinsics
from pycusfm.domain.enums.pipeline_mode import PipelineMode
from pycusfm.domain.enums.feature_type import FeatureType
from pycusfm.domain.exceptions.validation_errors import EntityValidationError


@pytest.fixture
def sample_project(temp_dir):
    """Create a sample SfM project."""
    input_dir = temp_dir / "input"
    input_dir.mkdir()
    workspace_dir = temp_dir / "workspace"
    workspace_dir.mkdir()

    return SfMProject(project_id="test_project",
                      input_directory=input_dir,
                      workspace_directory=workspace_dir,
                      mode=PipelineMode.ISAAC_ROBOTICS,
                      feature_type=FeatureType.SIFT)


@pytest.fixture
def sample_rig():
    """Create a sample camera rig."""
    intrinsics = CameraIntrinsics(fx=500.0,
                                  fy=500.0,
                                  cx=320.0,
                                  cy=240.0,
                                  width=640,
                                  height=480)
    camera = Camera(camera_id="cam0", intrinsics=intrinsics)
    rig = CameraRig(rig_id="rig0", cameras=[camera])
    return rig


class TestSfMProjectCreation:
    """Test SfM project creation."""

    def test_create_project(self, temp_dir):
        """Test creating a valid project."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        workspace_dir = temp_dir / "workspace"

        project = SfMProject(
            project_id="test_project",
            input_directory=input_dir,
            workspace_directory=workspace_dir,
            mode=PipelineMode.ISAAC_ROBOTICS,
        )

        assert project.project_id == "test_project"
        assert project.mode == PipelineMode.ISAAC_ROBOTICS
        assert project.feature_type == FeatureType.SIFT  # Default

    def test_project_requires_id(self, temp_dir):
        """Test that project ID cannot be empty."""
        with pytest.raises(ValueError, match="cannot be empty"):
            SfMProject(project_id="",
                       input_directory=temp_dir,
                       workspace_directory=temp_dir,
                       mode=PipelineMode.ISAAC_ROBOTICS)


class TestCameraRigManagement:
    """Test camera rig management."""

    def test_add_rig_to_project(self, sample_project, sample_rig):
        """Test adding a camera rig to project."""
        sample_project.add_camera_rig(sample_rig)

        assert sample_project.num_rigs == 1
        assert sample_project.has_rig("rig0")
        assert sample_project.get_rig("rig0") == sample_rig

    def test_cannot_add_duplicate_rig(self, sample_project, sample_rig):
        """Test that duplicate rig IDs are not allowed."""
        sample_project.add_camera_rig(sample_rig)

        with pytest.raises(ValueError, match="already exists"):
            sample_project.add_camera_rig(sample_rig)


class TestFrameManagement:
    """Test frame management."""

    def test_add_frame_to_project(self, sample_project, sample_rig,
                                   sample_image_path):
        """Test adding a frame to project."""
        sample_project.add_camera_rig(sample_rig)

        frame = FrameMetadata(frame_id="frame001",
                              camera_id="cam0",
                              image_path=sample_image_path,
                              timestamp_us=1000000)

        sample_project.add_frame(frame)

        assert sample_project.num_frames == 1
        assert sample_project.has_frame("frame001")
        assert sample_project.get_frame("frame001") == frame

    def test_get_frames_by_camera(self, sample_project, sample_rig,
                                   sample_image_path):
        """Test getting frames by camera ID."""
        sample_project.add_camera_rig(sample_rig)

        frame1 = FrameMetadata(frame_id="frame001",
                               camera_id="cam0",
                               image_path=sample_image_path,
                               timestamp_us=1000000)
        frame2 = FrameMetadata(frame_id="frame002",
                               camera_id="cam0",
                               image_path=sample_image_path,
                               timestamp_us=2000000)

        sample_project.add_frame(frame1)
        sample_project.add_frame(frame2)

        frames = sample_project.get_frames_by_camera("cam0")
        assert len(frames) == 2


class TestFeatureManagement:
    """Test feature management."""

    def test_add_feature_collection(self, sample_project, sample_rig,
                                     sample_image_path):
        """Test adding feature collection."""
        sample_project.add_camera_rig(sample_rig)

        frame = FrameMetadata(frame_id="frame001",
                              camera_id="cam0",
                              image_path=sample_image_path,
                              timestamp_us=1000000)
        sample_project.add_frame(frame)

        collection = FeatureCollection(frame_id="frame001",
                                        feature_type=FeatureType.SIFT)
        sample_project.add_feature_collection(collection)

        assert sample_project.num_frames_with_features == 1
        assert sample_project.has_features("frame001")
        assert sample_project.get_feature_collection(
            "frame001") == collection

    def test_cannot_add_features_for_nonexistent_frame(self, sample_project):
        """Test that features require existing frame."""
        collection = FeatureCollection(frame_id="nonexistent",
                                        feature_type=FeatureType.SIFT)

        with pytest.raises(ValueError, match="does not exist"):
            sample_project.add_feature_collection(collection)


class TestPoseGraphAndMapManagement:
    """Test pose graph and keypoint map management."""

    def test_initialize_pose_graph(self, sample_project):
        """Test initializing pose graph."""
        assert not sample_project.has_pose_graph()

        sample_project.initialize_pose_graph()

        assert sample_project.has_pose_graph()
        assert sample_project.pose_graph.project_id == "test_project"

    def test_initialize_keypoint_map(self, sample_project):
        """Test initializing keypoint map."""
        assert not sample_project.has_keypoint_map()

        sample_project.initialize_keypoint_map()

        assert sample_project.has_keypoint_map()
        assert sample_project.keypoint_map.project_id == "test_project"


class TestProjectStatistics:
    """Test project statistics."""

    def test_get_statistics(self, sample_project, sample_rig):
        """Test getting project statistics."""
        sample_project.add_camera_rig(sample_rig)
        sample_project.initialize_pose_graph()

        stats = sample_project.get_statistics()

        assert stats['project_id'] == "test_project"
        assert stats['num_rigs'] == 1
        assert stats['num_frames'] == 0
        assert stats['has_pose_graph'] is True
        assert stats['has_keypoint_map'] is False
