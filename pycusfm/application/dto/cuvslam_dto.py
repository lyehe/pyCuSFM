# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""cuVSLAM DTOs"""

from dataclasses import dataclass
from pathlib import Path
from ...domain.entities.sfm_project import SfMProject


@dataclass
class CuVSLAMRequest:
    """
    Input data for cuVSLAM use case.

    Attributes:
        project: SfM project containing frames to process
        output_directory: Where to write SLAM outputs
        frames_meta_file: Path to frames metadata JSON
        config_path: Path to cuVSLAM configuration
        use_slam_pose: Whether to use SLAM poses vs odometry poses
    """

    project: SfMProject
    output_directory: Path
    frames_meta_file: Path
    config_path: Path = None
    use_slam_pose: bool = True


@dataclass
class CuVSLAMResponse:
    """
    Output data from cuVSLAM use case.

    Attributes:
        slam_poses_file: Path to output SLAM poses (TUM format)
        odom_poses_file: Path to output odometry poses (TUM format)
        num_poses: Number of poses estimated
        execution_time_seconds: Time taken for SLAM
        success: Whether SLAM completed successfully
    """

    slam_poses_file: Path
    odom_poses_file: Path
    num_poses: int
    execution_time_seconds: float
    success: bool = True
