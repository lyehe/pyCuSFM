# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ExtractFeaturesUseCase"""

import pytest
from pathlib import Path
from pycusfm.application.use_cases.features.extract_features import ExtractFeaturesUseCase
from pycusfm.application.dto.feature_extraction_dto import (
    FeatureExtractionRequest,
)
from pycusfm.application.interfaces.config_repository import IConfigRepository
from pycusfm.domain.entities.sfm_project import SfMProject
from pycusfm.domain.entities.camera_rig import CameraRig, Camera
from pycusfm.domain.entities.frame_metadata import FrameMetadata
from pycusfm.domain.value_objects.camera_intrinsics import CameraIntrinsics
from pycusfm.domain.enums.feature_type import FeatureType
from pycusfm.domain.enums.pipeline_mode import PipelineMode
from tests.fixtures.mock_binary_executor import MockBinaryExecutor
from tests.fixtures.mock_file_repository import MockFileRepository
from tests.fixtures.mock_logger import MockLogger


class MockConfigRepository(IConfigRepository):
    """Simple mock config repository."""

    def load_config(self, config_name: str):
        return {}

    def get_config_path(self, config_name: str):
        return Path(f"/config/{config_name}")

    def config_exists(self, config_name: str):
        return True

    def list_configs(self):
        return []


@pytest.fixture
def mock_executor():
    """Create mock binary executor."""
    return MockBinaryExecutor(should_succeed=True)


@pytest.fixture
def mock_file_repo():
    """Create mock file repository."""
    repo = MockFileRepository()
    # Simulate directories exist
    repo.ensure_directory_exists(Path("/input"))
    repo.ensure_directory_exists(Path("/workspace"))
    return repo


@pytest.fixture
def mock_config_repo():
    """Create mock config repository."""
    return MockConfigRepository()


@pytest.fixture
def mock_logger():
    """Create mock logger."""
    return MockLogger()


@pytest.fixture
def sample_project(tmp_path):
    """Create a sample SfM project for testing."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create project
    project = SfMProject(
        project_id="test_project",
        input_directory=input_dir,
        workspace_directory=workspace,
        mode=PipelineMode.ISAAC_ROBOTICS,
        feature_type=FeatureType.SIFT,
    )

    # Add a camera rig
    intrinsics = CameraIntrinsics(
        fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480
    )
    camera = Camera(camera_id="cam0", intrinsics=intrinsics)
    rig = CameraRig(rig_id="rig0", cameras=[camera])
    project.add_camera_rig(rig)

    # Add a frame
    image_path = input_dir / "frame001.jpg"
    image_path.touch()
    frame = FrameMetadata(
        frame_id="frame001",
        camera_id="cam0",
        image_path=image_path,
        timestamp_us=1000000,
    )
    project.add_frame(frame)

    return project


class TestExtractFeaturesUseCase:
    """Test ExtractFeaturesUseCase."""

    def test_execute_success(
        self,
        sample_project,
        mock_executor,
        mock_file_repo,
        mock_config_repo,
        mock_logger,
    ):
        """Test successful feature extraction."""
        # Arrange
        use_case = ExtractFeaturesUseCase(
            binary_executor=mock_executor,
            file_repository=mock_file_repo,
            config_repository=mock_config_repo,
            logger=mock_logger,
        )

        request = FeatureExtractionRequest(
            project=sample_project,
            output_directory=sample_project.workspace_directory / "features",
            feature_type=FeatureType.SIFT,
            max_keypoints=1000,
        )

        # Act
        result = use_case.execute(request)

        # Assert
        assert result.success
        assert result.data is not None
        assert result.data.features_directory == request.output_directory
        assert mock_executor.execute_called
        assert "feature_extractor_main" in mock_executor.last_binary_name
        assert len(mock_logger.info_messages) > 0

    def test_execute_fails_if_binary_fails(
        self,
        sample_project,
        mock_file_repo,
        mock_config_repo,
        mock_logger,
    ):
        """Test that use case fails if binary execution fails."""
        # Arrange
        failing_executor = MockBinaryExecutor(should_succeed=False)

        use_case = ExtractFeaturesUseCase(
            binary_executor=failing_executor,
            file_repository=mock_file_repo,
            config_repository=mock_config_repo,
            logger=mock_logger,
        )

        request = FeatureExtractionRequest(
            project=sample_project,
            output_directory=sample_project.workspace_directory / "features",
            feature_type=FeatureType.SIFT,
        )

        # Act
        result = use_case.execute(request)

        # Assert
        assert not result.success
        assert result.error is not None
        assert "failed" in result.error.lower()

    def test_validates_max_keypoints(
        self,
        sample_project,
        mock_executor,
        mock_file_repo,
        mock_config_repo,
        mock_logger,
    ):
        """Test that invalid max_keypoints is rejected."""
        # Arrange
        use_case = ExtractFeaturesUseCase(
            binary_executor=mock_executor,
            file_repository=mock_file_repo,
            config_repository=mock_config_repo,
            logger=mock_logger,
        )

        request = FeatureExtractionRequest(
            project=sample_project,
            output_directory=sample_project.workspace_directory / "features",
            feature_type=FeatureType.SIFT,
            max_keypoints=-5,  # Invalid!
        )

        # Act
        result = use_case.execute(request)

        # Assert
        assert not result.success
        assert "validation" in result.error.lower()
