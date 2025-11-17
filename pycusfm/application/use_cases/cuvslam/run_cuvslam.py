# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Run cuVSLAM Use Case"""

from pathlib import Path
from typing import List
from ..base_use_case import BaseUseCase, UseCaseResult
from ...interfaces.binary_executor import IBinaryExecutor
from ...interfaces.file_repository import IFileRepository
from ...interfaces.config_repository import IConfigRepository
from ...interfaces.logger import ILogger
from ...dto.cuvslam_dto import CuVSLAMRequest, CuVSLAMResponse


class RunCuVSLAMUseCase(BaseUseCase[CuVSLAMRequest, CuVSLAMResponse]):
    """
    Use case for running cuVSLAM (CUDA Visual SLAM).

    This use case:
    1. Validates input (frames metadata exists)
    2. Prepares output directory
    3. Builds command arguments for cuvslam_api_launcher binary
    4. Executes cuVSLAM via IBinaryExecutor
    5. Parses output poses
    6. Returns result with pose file paths

    cuVSLAM provides initial camera pose estimates that are later
    refined by pose graph optimization and bundle adjustment.
    """

    def __init__(
        self,
        binary_executor: IBinaryExecutor,
        file_repository: IFileRepository,
        config_repository: IConfigRepository,
        logger: ILogger,
    ):
        """
        Initialize use case with dependencies.

        Args:
            binary_executor: For executing cuvslam_api_launcher binary
            file_repository: For file operations
            config_repository: For loading cuVSLAM config
            logger: For logging
        """
        self.binary_executor = binary_executor
        self.file_repository = file_repository
        self.config_repository = config_repository
        self.logger = logger

    def execute(
            self, request: CuVSLAMRequest) -> UseCaseResult[CuVSLAMResponse]:
        """
        Execute cuVSLAM.

        Args:
            request: cuVSLAM parameters

        Returns:
            UseCaseResult with CuVSLAMResponse or error
        """
        try:
            # Validate request
            self._validate_request(request)
            self.logger.info("Starting cuVSLAM...")

            # Prepare output directory
            self.file_repository.ensure_directory_exists(
                request.output_directory)

            # Output file paths
            slam_poses_file = request.output_directory / "slam_poses.tum"
            odom_poses_file = request.output_directory / "odom_poses.tum"

            # Build command arguments
            args = self._build_arguments(request, slam_poses_file,
                                         odom_poses_file)

            # Execute binary
            self.logger.info("Executing cuvslam_api_launcher binary...")
            result = self.binary_executor.execute(
                binary_name="cuvslam_api_launcher",
                arguments=args,
            )

            if not result.success:
                return UseCaseResult.fail(
                    error=f"cuVSLAM failed: {result.stderr}",
                    exit_code=result.exit_code,
                    execution_time=result.execution_time_seconds,
                )

            # Count poses in output
            num_poses = self._count_poses(slam_poses_file)

            response = CuVSLAMResponse(
                slam_poses_file=slam_poses_file,
                odom_poses_file=odom_poses_file,
                num_poses=num_poses,
                execution_time_seconds=result.execution_time_seconds,
                success=True,
            )

            self.logger.info(
                f"cuVSLAM completed: {num_poses} poses estimated "
                f"in {result.execution_time_seconds:.2f}s")

            return UseCaseResult.ok(
                data=response,
                execution_time=result.execution_time_seconds,
            )

        except ValueError as e:
            self.logger.error(f"Validation error: {e}")
            return UseCaseResult.fail(error=f"Validation error: {e}")
        except Exception as e:
            self.logger.exception("cuVSLAM failed", exc_info=e)
            return UseCaseResult.fail(error=f"Unexpected error: {e}")

    def _validate_request(self, request: CuVSLAMRequest) -> None:
        """
        Validate cuVSLAM request.

        Args:
            request: Request to validate

        Raises:
            ValueError: If request is invalid
        """
        super()._validate_request(request)

        if not self.file_repository.file_exists(request.frames_meta_file):
            raise ValueError(
                f"Frames metadata file does not exist: {request.frames_meta_file}"
            )

        if request.project.num_frames == 0:
            raise ValueError("Project has no frames to process")

    def _build_arguments(self, request: CuVSLAMRequest,
                         slam_poses_file: Path,
                         odom_poses_file: Path) -> List[str]:
        """
        Build command-line arguments for cuvslam_api_launcher.

        Args:
            request: cuVSLAM request
            slam_poses_file: Output path for SLAM poses
            odom_poses_file: Output path for odometry poses

        Returns:
            List of command-line arguments
        """
        args = [
            "--frames_meta",
            str(request.frames_meta_file),
            "--slam_poses_output",
            str(slam_poses_file),
            "--odom_poses_output",
            str(odom_poses_file),
        ]

        # Optional: config file
        if request.config_path and request.config_path.exists():
            args.extend(["--config", str(request.config_path)])

        # Optional: pose type preference
        if request.use_slam_pose:
            args.append("--use_slam_pose")

        return args

    def _count_poses(self, poses_file: Path) -> int:
        """
        Count number of poses in TUM format file.

        TUM format: timestamp tx ty tz qx qy qz qw

        Args:
            poses_file: Path to poses file

        Returns:
            Number of poses
        """
        try:
            if not self.file_repository.file_exists(poses_file):
                return 0

            content = self.file_repository.read_text(poses_file)
            lines = [
                line.strip() for line in content.split('\n')
                if line.strip() and not line.startswith('#')
            ]
            return len(lines)

        except Exception as e:
            self.logger.warning(f"Could not count poses: {e}")
            return 0
