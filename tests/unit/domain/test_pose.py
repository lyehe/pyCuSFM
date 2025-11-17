# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Pose value object"""

import pytest
import numpy as np
from pycusfm.domain.value_objects.pose import Pose


class TestPoseCreation:
    """Test pose creation and validation."""

    def test_create_valid_pose(self, sample_position, normalized_quaternion):
        """Test creating a valid pose."""
        pose = Pose(position=sample_position,
                    orientation=normalized_quaternion,
                    timestamp_us=1000000)

        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 3.0
        assert pose.qw == 1.0
        assert pose.timestamp_us == 1000000

    def test_pose_requires_normalized_quaternion(self, sample_position):
        """Test that quaternion must be normalized."""
        with pytest.raises(ValueError, match="must be normalized"):
            Pose(position=sample_position,
                 orientation=(1.0, 1.0, 0.0, 0.0),  # Not normalized
                 timestamp_us=1000000)

    def test_pose_requires_3d_position(self, normalized_quaternion):
        """Test that position must be 3D."""
        with pytest.raises(ValueError, match="must be a 3-tuple"):
            Pose(position=(1.0, 2.0),  # Only 2D
                 orientation=normalized_quaternion,
                 timestamp_us=1000000)

    def test_pose_requires_4d_quaternion(self, sample_position):
        """Test that quaternion must be 4D."""
        with pytest.raises(ValueError, match="must be a 4-tuple"):
            Pose(position=sample_position,
                 orientation=(1.0, 0.0, 0.0),  # Only 3D
                 timestamp_us=1000000)

    def test_pose_requires_non_negative_timestamp(
            self, sample_position, normalized_quaternion):
        """Test that timestamp must be non-negative."""
        with pytest.raises(ValueError, match="must be non-negative"):
            Pose(position=sample_position,
                 orientation=normalized_quaternion,
                 timestamp_us=-1000)


class TestPoseTransformations:
    """Test pose transformations."""

    def test_to_matrix(self, sample_position, normalized_quaternion):
        """Test conversion to transformation matrix."""
        pose = Pose(position=sample_position,
                    orientation=normalized_quaternion,
                    timestamp_us=1000000)

        matrix = pose.to_matrix()

        assert matrix.shape == (4, 4)
        assert matrix[3, 3] == 1.0
        # Check position is in last column
        assert matrix[0, 3] == 1.0
        assert matrix[1, 3] == 2.0
        assert matrix[2, 3] == 3.0

    def test_from_matrix(self):
        """Test creating pose from matrix."""
        # Create identity transformation at (1, 2, 3)
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]

        pose = Pose.from_matrix(T, timestamp_us=1000000)

        assert pose.x == pytest.approx(1.0)
        assert pose.y == pytest.approx(2.0)
        assert pose.z == pytest.approx(3.0)
        assert pose.timestamp_us == 1000000

    def test_pose_inverse(self, sample_position, normalized_quaternion):
        """Test pose inverse."""
        pose = Pose(position=sample_position,
                    orientation=normalized_quaternion,
                    timestamp_us=1000000)

        inverse = pose.inverse()

        # Check that pose * inverse ≈ identity
        T_original = pose.to_matrix()
        T_inverse = inverse.to_matrix()
        T_result = T_original @ T_inverse

        assert np.allclose(T_result, np.eye(4))


class TestPoseImmutability:
    """Test that Pose is immutable."""

    def test_pose_is_frozen(self, sample_position, normalized_quaternion):
        """Test that pose attributes cannot be modified."""
        pose = Pose(position=sample_position,
                    orientation=normalized_quaternion,
                    timestamp_us=1000000)

        with pytest.raises(Exception):  # dataclass frozen raises FrozenInstanceError
            pose.position = (0.0, 0.0, 0.0)
