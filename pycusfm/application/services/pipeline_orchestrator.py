# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Pipeline Orchestrator Service"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
from ...domain.entities.sfm_project import SfMProject
from ..dto.pipeline_config import PipelineConfig
from ..dto.feature_extraction_dto import FeatureExtractionRequest
from ..dto.cuvslam_dto import CuVSLAMRequest
from ..use_cases.features.extract_features import ExtractFeaturesUseCase
from ..use_cases.cuvslam.run_cuvslam import RunCuVSLAMUseCase
from ..use_cases.features.match_features import (
    MatchFeaturesUseCase,
    MatchFeaturesRequest,
)
from ..interfaces.logger import ILogger


@dataclass
class PipelineStepResult:
    """Result from a single pipeline step."""

    step_name: str
    success: bool
    error: Optional[str] = None
    execution_time_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Overall pipeline execution result."""

    success: bool
    project: SfMProject
    steps: List[PipelineStepResult] = field(default_factory=list)
    total_time_seconds: float = 0.0
    error: Optional[str] = None

    @property
    def completed_steps(self) -> List[str]:
        """Get list of successfully completed step names."""
        return [
            step.step_name for step in self.steps if step.success
        ]

    @property
    def failed_step(self) -> Optional[PipelineStepResult]:
        """Get the first failed step, if any."""
        for step in self.steps:
            if not step.success:
                return step
        return None


class PipelineOrchestrator:
    """
    Orchestrates the complete SfM pipeline.

    This service coordinates all use cases in the correct order:
    1. cuVSLAM (optional) - Initial pose estimation
    2. Feature Extraction - Extract features from images
    3. BoW Vocabulary - Generate bag-of-words vocabulary
    4. Associations - Generate image associations
    5. Pose Graph - Optimize pose graph
    6. Feature Matching - Match features between pairs
    7. Keypoint Mapping - 3D reconstruction with bundle adjustment
    8. Map Conversion - Convert to COLMAP format

    This demonstrates the Open/Closed Principle: we can add new steps
    without modifying existing code.
    """

    def __init__(
        self,
        cuvslam_use_case: Optional[RunCuVSLAMUseCase] = None,
        extract_features_use_case: Optional[ExtractFeaturesUseCase] = None,
        match_features_use_case: Optional[MatchFeaturesUseCase] = None,
        # Add other use cases as needed
        logger: Optional[ILogger] = None,
    ):
        """
        Initialize orchestrator with use cases.

        Args:
            cuvslam_use_case: cuVSLAM use case (optional)
            extract_features_use_case: Feature extraction use case
            match_features_use_case: Feature matching use case
            logger: Logger for orchestration
        """
        self.cuvslam_use_case = cuvslam_use_case
        self.extract_features_use_case = extract_features_use_case
        self.match_features_use_case = match_features_use_case
        self.logger = logger

    def run_pipeline(
        self,
        project: SfMProject,
        config: PipelineConfig,
    ) -> PipelineResult:
        """
        Run the complete SfM pipeline.

        Args:
            project: SfM project to process
            config: Pipeline configuration

        Returns:
            PipelineResult with overall success and step results
        """
        import time

        start_time = time.time()
        result = PipelineResult(success=True, project=project)

        try:
            # Log pipeline start
            if self.logger:
                self.logger.info(
                    f"Starting SfM pipeline for project {project.project_id}"
                )
                self.logger.info(f"Mode: {config.mode.value}")
                self.logger.info(
                    f"Feature type: {config.feature_type.value}")

            # Step 1: cuVSLAM (optional)
            if not config.skip_cuvslam and self.cuvslam_use_case:
                step_result = self._run_cuvslam(project, config)
                result.steps.append(step_result)
                if not step_result.success:
                    result.success = False
                    result.error = f"cuVSLAM failed: {step_result.error}"
                    return result

            # Step 2: Feature Extraction
            if not config.skip_feature_extractor and self.extract_features_use_case:
                step_result = self._run_feature_extraction(project, config)
                result.steps.append(step_result)
                if not step_result.success:
                    result.success = False
                    result.error = f"Feature extraction failed: {step_result.error}"
                    return result

            # Step 3: Feature Matching
            if not config.skip_matcher and self.match_features_use_case:
                step_result = self._run_feature_matching(project, config)
                result.steps.append(step_result)
                if not step_result.success:
                    result.success = False
                    result.error = f"Feature matching failed: {step_result.error}"
                    return result

            # TODO: Add remaining steps (BoW, associations, pose graph, mapping, conversion)

            # Pipeline completed successfully
            result.total_time_seconds = time.time() - start_time

            if self.logger:
                self.logger.info(
                    f"Pipeline completed successfully in {result.total_time_seconds:.2f}s"
                )
                self.logger.info(
                    f"Completed steps: {', '.join(result.completed_steps)}")

            return result

        except Exception as e:
            result.success = False
            result.error = f"Pipeline failed: {e}"
            result.total_time_seconds = time.time() - start_time

            if self.logger:
                self.logger.exception("Pipeline failed", exc_info=e)

            return result

    def _run_cuvslam(self, project: SfMProject,
                     config: PipelineConfig) -> PipelineStepResult:
        """Run cuVSLAM step."""
        if self.logger:
            self.logger.info("Running step: cuVSLAM")

        frames_meta_file = project.workspace_directory / "frames_meta.json"

        request = CuVSLAMRequest(
            project=project,
            output_directory=project.workspace_directory / "cuvslam_output",
            frames_meta_file=frames_meta_file,
            use_slam_pose=config.use_cuvslam_slam_pose,
        )

        result = self.cuvslam_use_case.execute(request)

        return PipelineStepResult(
            step_name="cuVSLAM",
            success=result.success,
            error=result.error,
            execution_time_seconds=result.metadata.get(
                'execution_time', 0.0),
            metadata=result.metadata,
        )

    def _run_feature_extraction(
            self, project: SfMProject,
            config: PipelineConfig) -> PipelineStepResult:
        """Run feature extraction step."""
        if self.logger:
            self.logger.info("Running step: Feature Extraction")

        request = FeatureExtractionRequest(
            project=project,
            output_directory=project.workspace_directory / "keyframes",
            feature_type=config.feature_type,
            max_keypoints=config.max_keypoints,
            batch_size=config.feature_extractor_batch_size,
            enable_debug=config.enable_debug,
        )

        result = self.extract_features_use_case.execute(request)

        return PipelineStepResult(
            step_name="Feature Extraction",
            success=result.success,
            error=result.error,
            execution_time_seconds=result.metadata.get(
                'execution_time', 0.0),
            metadata=result.metadata,
        )

    def _run_feature_matching(
            self, project: SfMProject,
            config: PipelineConfig) -> PipelineStepResult:
        """Run feature matching step."""
        if self.logger:
            self.logger.info("Running step: Feature Matching")

        features_dir = project.workspace_directory / "keyframes"
        matches_dir = project.workspace_directory / "matches"
        match_pairs_file = project.workspace_directory / "associations" / "match_pairs.txt"

        request = MatchFeaturesRequest(
            features_directory=features_dir,
            output_directory=matches_dir,
            match_pairs_file=match_pairs_file,
            batch_size=config.feature_matching_batch_size,
        )

        result = self.match_features_use_case.execute(request)

        return PipelineStepResult(
            step_name="Feature Matching",
            success=result.success,
            error=result.error,
            execution_time_seconds=result.metadata.get(
                'execution_time', 0.0),
            metadata=result.metadata,
        )
