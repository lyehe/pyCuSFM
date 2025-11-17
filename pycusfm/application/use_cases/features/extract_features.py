# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Extract Features Use Case"""

from pathlib import Path
from typing import List
from ..base_use_case import BaseUseCase, UseCaseResult
from ...interfaces.binary_executor import IBinaryExecutor
from ...interfaces.file_repository import IFileRepository
from ...interfaces.config_repository import IConfigRepository
from ...interfaces.logger import ILogger
from ...dto.feature_extraction_dto import (
    FeatureExtractionRequest,
    FeatureExtractionResponse,
)
from ....domain.entities.feature_collection import FeatureCollection
from ....domain.enums.feature_type import FeatureType


class ExtractFeaturesUseCase(
        BaseUseCase[FeatureExtractionRequest, FeatureExtractionResponse]):
    """
    Use case for extracting visual features from images.

    This use case:
    1. Validates input (project has frames)
    2. Prepares output directory
    3. Builds command arguments for feature_extractor_main binary
    4. Executes feature extraction via IBinaryExecutor
    5. Updates project with feature collections
    6. Returns result

    Dependencies are injected via constructor (Dependency Inversion Principle).
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
            binary_executor: For executing feature_extractor_main binary
            file_repository: For file operations
            config_repository: For loading feature extraction config
            logger: For logging
        """
        self.binary_executor = binary_executor
        self.file_repository = file_repository
        self.config_repository = config_repository
        self.logger = logger

    def execute(
        self, request: FeatureExtractionRequest
    ) -> UseCaseResult[FeatureExtractionResponse]:
        """
        Execute feature extraction.

        Args:
            request: Feature extraction parameters

        Returns:
            UseCaseResult with FeatureExtractionResponse or error
        """
        try:
            # Validate request
            self._validate_request(request)
            self.logger.info(
                f"Starting feature extraction: {request.feature_type.value}")

            # Prepare output directory
            self.file_repository.ensure_directory_exists(
                request.output_directory)

            # Build command arguments
            args = self._build_arguments(request)

            # Execute binary
            self.logger.info(f"Executing feature_extractor_main binary...")
            result = self.binary_executor.execute(
                binary_name="feature_extractor_main",
                arguments=args,
            )

            if not result.success:
                return UseCaseResult.fail(
                    error=f"Feature extraction failed: {result.stderr}",
                    exit_code=result.exit_code,
                    execution_time=result.execution_time_seconds,
                )

            # Parse results
            num_processed = self._count_processed_images(
                request.output_directory)

            response = FeatureExtractionResponse(
                features_directory=request.output_directory,
                num_images_processed=num_processed,
                total_keypoints=0,  # TODO: Parse from output
                average_keypoints_per_image=0.0,
                execution_time_seconds=result.execution_time_seconds,
            )

            self.logger.info(
                f"Feature extraction completed: {num_processed} images processed "
                f"in {result.execution_time_seconds:.2f}s")

            return UseCaseResult.ok(
                data=response,
                execution_time=result.execution_time_seconds,
            )

        except ValueError as e:
            self.logger.error(f"Validation error: {e}")
            return UseCaseResult.fail(error=f"Validation error: {e}")
        except Exception as e:
            self.logger.exception("Feature extraction failed", exc_info=e)
            return UseCaseResult.fail(error=f"Unexpected error: {e}")

    def _validate_request(self, request: FeatureExtractionRequest) -> None:
        """
        Validate feature extraction request.

        Args:
            request: Request to validate

        Raises:
            ValueError: If request is invalid
        """
        super()._validate_request(request)

        if not request.project.input_directory.exists():
            raise ValueError(
                f"Input directory does not exist: {request.project.input_directory}"
            )

        if request.project.num_frames == 0:
            raise ValueError("Project has no frames to process")

        if request.max_keypoints < -1:
            raise ValueError(
                f"max_keypoints must be >= -1, got {request.max_keypoints}")

        if request.batch_size < 0:
            raise ValueError(
                f"batch_size must be >= 0, got {request.batch_size}")

    def _build_arguments(
        self, request: FeatureExtractionRequest
    ) -> List[str]:
        """
        Build command-line arguments for feature_extractor_main.

        Args:
            request: Feature extraction request

        Returns:
            List of command-line arguments
        """
        args = [
            "--input_dir",
            str(request.project.input_directory),
            "--output_dir",
            str(request.output_directory),
            "--feature_type",
            request.feature_type.value,
        ]

        # Optional: max keypoints
        if request.max_keypoints > 0:
            args.extend(["--max_keypoints", str(request.max_keypoints)])

        # Optional: batch size
        if request.batch_size > 0:
            args.extend(["--batch_size", str(request.batch_size)])

        # Optional: config file
        if request.config_path and request.config_path.exists():
            args.extend(["--config", str(request.config_path)])

        # Optional: debug mode
        if request.enable_debug:
            args.append("--debug")

        return args

    def _count_processed_images(self, output_dir: Path) -> int:
        """
        Count number of images that were processed.

        Args:
            output_dir: Directory containing extracted features

        Returns:
            Number of processed images
        """
        # Feature files typically have .feat or .bin extension
        # This is a simplified version - actual implementation
        # would parse feature file metadata
        try:
            feature_files = self.file_repository.list_files(
                output_dir, pattern="*.feat")
            if not feature_files:
                # Try alternative extension
                feature_files = self.file_repository.list_files(
                    output_dir, pattern="*.bin")
            return len(feature_files)
        except Exception as e:
            self.logger.warning(
                f"Could not count processed images: {e}")
            return 0
