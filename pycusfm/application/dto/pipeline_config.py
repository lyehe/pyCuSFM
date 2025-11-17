# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Pipeline Configuration DTO"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from ...domain.enums.feature_type import FeatureType
from ...domain.enums.pipeline_mode import PipelineMode
from ...domain.enums.coordinate_frame import CoordinateFrame


@dataclass
class PipelineConfig:
    """
    Configuration for the entire SfM pipeline.

    This DTO encapsulates all configuration parameters needed
    to run the pipeline, avoiding the 40+ parameter problem.
    """

    # Directory configuration
    binary_dir: Path
    config_dir: Path
    model_dir: Optional[Path] = None
    mask_dir: Optional[Path] = None

    # Feature configuration
    feature_type: FeatureType = FeatureType.SIFT
    max_keypoints: int = -1  # -1 = use config default
    feature_extractor_batch_size: int = 0  # 0 = auto
    feature_matching_batch_size: int = 0  # 0 = auto

    # Pipeline mode
    mode: PipelineMode = PipelineMode.ISAAC_ROBOTICS
    coordinate_frame: CoordinateFrame = CoordinateFrame.CAMERA_FRAME

    # Skip flags
    skip_cuvslam: bool = False
    skip_feature_extractor: bool = False
    skip_vocab_generator: bool = False
    skip_pose_graph: bool = False
    skip_matcher: bool = False
    skip_mapper: bool = False
    skip_map_convertor: bool = False
    skip_data_association: bool = False

    # Optimization settings
    optimize_extrinsics: bool = False
    use_vehicle_trajectory: bool = False
    downsampling_matches: bool = True

    # Multi-track settings
    multi_track_input: bool = False
    anchor_track: str = ""
    global_localize_succ_sample: int = 10
    skip_track_global_transform: bool = False

    # Frame selection
    min_inter_frame_distance: float = 0.5  # meters
    min_inter_frame_rotation_degrees: float = 5.0  # degrees

    # Stereo settings
    sample_sync_threshold_microseconds: int = 0
    stereo_pair_non_baseline_max_distance: float = 0.0

    # Rolling shutter
    use_roll_shutter_correction: bool = False

    # Debug settings
    enable_debug: bool = False
    debug_interval: int = 500
    dry_run: bool = False

    # Performance
    num_threads: int = 1

    # Previous workspace (patch mode)
    previous_cusfm_ws: str = ""
    previous_raw_image_dir: str = ""

    # Output settings
    export_pose_in_vehicle_frame: bool = True
    use_cuvslam_slam_pose: bool = True

    # Environment
    add_tensorrt_path: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.optimize_extrinsics and self.coordinate_frame != CoordinateFrame.VEHICLE_RIG:
            raise ValueError(
                "optimize_extrinsics requires coordinate_frame=VEHICLE_RIG"
            )

        if self.num_threads < 1:
            raise ValueError(f"num_threads must be >= 1, got {self.num_threads}")

    @property
    def is_av_mode(self) -> bool:
        """Check if pipeline is in AV mode."""
        return self.mode == PipelineMode.AUTONOMOUS_VEHICLE

    @property
    def is_multi_track(self) -> bool:
        """Check if pipeline handles multiple tracks."""
        return self.mode == PipelineMode.MULTI_TRACK or self.multi_track_input

    @property
    def requires_tensorrt(self) -> bool:
        """Check if TensorRT is required."""
        return self.feature_type.requires_tensorrt
