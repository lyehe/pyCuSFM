# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Match Features Use Case"""

from ..base_use_case import BaseUseCase, UseCaseResult
from ...interfaces.binary_executor import IBinaryExecutor
from ...interfaces.file_repository import IFileRepository
from ...interfaces.logger import ILogger
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MatchFeaturesRequest:
    """Request for feature matching."""

    features_directory: Path
    output_directory: Path
    match_pairs_file: Path
    batch_size: int = 0


@dataclass
class MatchFeaturesResponse:
    """Response from feature matching."""

    matches_directory: Path
    num_pairs_matched: int
    execution_time_seconds: float


class MatchFeaturesUseCase(
        BaseUseCase[MatchFeaturesRequest, MatchFeaturesResponse]):
    """
    Use case for matching features between image pairs.

    Executes feature_matcher_main binary to compute feature correspondences
    using LightGlue or other matchers.
    """

    def __init__(
        self,
        binary_executor: IBinaryExecutor,
        file_repository: IFileRepository,
        logger: ILogger,
    ):
        self.binary_executor = binary_executor
        self.file_repository = file_repository
        self.logger = logger

    def execute(
            self,
            request: MatchFeaturesRequest) -> UseCaseResult[MatchFeaturesResponse]:
        """Execute feature matching."""
        try:
            self._validate_request(request)
            self.logger.info("Starting feature matching...")

            self.file_repository.ensure_directory_exists(
                request.output_directory)

            args = [
                "--features_dir",
                str(request.features_directory),
                "--output_dir",
                str(request.output_directory),
                "--match_pairs",
                str(request.match_pairs_file),
            ]

            if request.batch_size > 0:
                args.extend(["--batch_size", str(request.batch_size)])

            result = self.binary_executor.execute(
                binary_name="feature_matcher_main",
                arguments=args,
            )

            if not result.success:
                return UseCaseResult.fail(
                    error=f"Feature matching failed: {result.stderr}")

            response = MatchFeaturesResponse(
                matches_directory=request.output_directory,
                num_pairs_matched=0,  # TODO: Parse from output
                execution_time_seconds=result.execution_time_seconds,
            )

            return UseCaseResult.ok(data=response)

        except Exception as e:
            self.logger.exception("Feature matching failed", exc_info=e)
            return UseCaseResult.fail(error=str(e))
