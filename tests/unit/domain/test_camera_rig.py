# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CameraRig entity"""

import pytest
from pycusfm.domain.entities.camera_rig import CameraRig, Camera
from pycusfm.domain.value_objects.camera_intrinsics import CameraIntrinsics
from pycusfm.domain.value_objects.pose import Pose
from pycusfm.domain.exceptions.domain_exceptions import InvalidCameraConfigurationException


@pytest.fixture
def sample_intrinsics():
    """Create sample camera intrinsics."""
    return CameraIntrinsics(fx=500.0,
                            fy=500.0,
                            cx=320.0,
                            cy=240.0,
                            width=640,
                            height=480)


@pytest.fixture
def sample_camera(sample_intrinsics):
    """Create a sample camera."""
    return Camera(camera_id="cam0", intrinsics=sample_intrinsics)


@pytest.fixture
def sample_extrinsics():
    """Create sample extrinsics."""
    return Pose(position=(0.12, 0.0, 0.0),  # 12cm baseline
                orientation=(1.0, 0.0, 0.0, 0.0),
                timestamp_us=0,
                frame="rig_frame")


class TestCamera:
    """Test Camera entity."""

    def test_create_camera(self, sample_intrinsics):
        """Test creating a camera."""
        camera = Camera(camera_id="cam0", intrinsics=sample_intrinsics)

        assert camera.camera_id == "cam0"
        assert camera.intrinsics == sample_intrinsics
        assert not camera.has_extrinsics()

    def test_camera_with_extrinsics(self, sample_intrinsics,
                                     sample_extrinsics):
        """Test camera with extrinsics."""
        camera = Camera(camera_id="cam0",
                        intrinsics=sample_intrinsics,
                        extrinsics=sample_extrinsics)

        assert camera.has_extrinsics()
        assert camera.extrinsics == sample_extrinsics

    def test_camera_requires_id(self, sample_intrinsics):
        """Test that camera ID cannot be empty."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Camera(camera_id="", intrinsics=sample_intrinsics)


class TestCameraRig:
    """Test CameraRig entity."""

    def test_create_empty_rig(self):
        """Test creating an empty rig."""
        rig = CameraRig(rig_id="rig0")

        assert rig.rig_id == "rig0"
        assert rig.num_cameras == 0
        assert not rig.is_stereo

    def test_add_camera_to_rig(self, sample_camera):
        """Test adding a camera to rig."""
        rig = CameraRig(rig_id="rig0")
        rig.add_camera(sample_camera)

        assert rig.num_cameras == 1
        assert rig.has_camera("cam0")
        assert rig.get_camera("cam0") == sample_camera

    def test_cannot_add_duplicate_camera(self, sample_camera):
        """Test that duplicate camera IDs are not allowed."""
        rig = CameraRig(rig_id="rig0")
        rig.add_camera(sample_camera)

        with pytest.raises(ValueError, match="already exists"):
            rig.add_camera(sample_camera)

    def test_get_nonexistent_camera(self):
        """Test getting a camera that doesn't exist."""
        rig = CameraRig(rig_id="rig0")

        assert rig.get_camera("nonexistent") is None
        assert not rig.has_camera("nonexistent")


class TestStereoRigValidation:
    """Test stereo rig validation."""

    def test_stereo_rig_requires_two_cameras(self, sample_intrinsics):
        """Test that stereo rig must have exactly 2 cameras."""
        camera1 = Camera(camera_id="left", intrinsics=sample_intrinsics)

        rig = CameraRig(rig_id="stereo1",
                        cameras=[camera1],
                        is_stereo=True,
                        baseline=0.12)

        with pytest.raises(InvalidCameraConfigurationException,
                           match="must have exactly 2 cameras"):
            rig.validate_stereo_configuration()

    def test_stereo_rig_requires_positive_baseline(self, sample_intrinsics,
                                                    sample_extrinsics):
        """Test that stereo rig must have baseline > 0."""
        camera1 = Camera(camera_id="left",
                         intrinsics=sample_intrinsics,
                         extrinsics=sample_extrinsics)
        camera2 = Camera(camera_id="right",
                         intrinsics=sample_intrinsics,
                         extrinsics=sample_extrinsics)

        rig = CameraRig(rig_id="stereo1",
                        cameras=[camera1, camera2],
                        is_stereo=True,
                        baseline=0.0)  # Invalid!

        with pytest.raises(InvalidCameraConfigurationException,
                           match="positive baseline"):
            rig.validate_stereo_configuration()

    def test_valid_stereo_rig(self, sample_intrinsics, sample_extrinsics):
        """Test a valid stereo rig configuration."""
        camera1 = Camera(camera_id="left",
                         intrinsics=sample_intrinsics,
                         extrinsics=sample_extrinsics)
        camera2 = Camera(camera_id="right",
                         intrinsics=sample_intrinsics,
                         extrinsics=sample_extrinsics)

        rig = CameraRig(rig_id="stereo1",
                        cameras=[camera1, camera2],
                        is_stereo=True,
                        baseline=0.12)

        # Should not raise
        rig.validate_stereo_configuration()
        rig.validate()
