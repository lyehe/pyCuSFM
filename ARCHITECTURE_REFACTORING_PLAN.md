# PyCuSFM Architecture Refactoring Plan

**Version:** 1.0
**Date:** 2025-11-17
**Status:** Proposed

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Current Architecture Analysis](#current-architecture-analysis)
3. [SOLID Principles Violations](#solid-principles-violations)
4. [Proposed Clean Architecture](#proposed-clean-architecture)
5. [Detailed Refactoring Plan](#detailed-refactoring-plan)
6. [Migration Strategy](#migration-strategy)
7. [Testing Strategy](#testing-strategy)
8. [Timeline & Phases](#timeline--phases)

---

## 1. Executive Summary

### Current State
PyCuSFM is a CUDA-accelerated Structure-from-Motion framework with:
- **~2000 lines** of Python orchestration code
- **22 compiled binaries** for heavy computation
- **Monolithic design** with tight coupling
- **No test structure**
- **Mixed concerns** across classes

### Proposed State
A well-architected system following:
- **SOLID principles** for maintainability
- **Clean Architecture** for separation of concerns
- **Domain-Driven Design** for clear business logic
- **Testability** with >80% code coverage
- **Extensibility** for future feature additions

### Key Benefits
- **Maintainability**: Easier to understand and modify
- **Testability**: Comprehensive unit and integration tests
- **Extensibility**: Add new features without breaking existing code
- **Reusability**: Share components across different contexts
- **Quality**: Better error handling, logging, and monitoring

---

## 2. Current Architecture Analysis

### 2.1 File Structure (Current)
```
pycusfm/
├── __init__.py                 # Package exports
├── constants.py                # Constants (34 lines)
├── cusfm_runner.py            # MONOLITHIC (1088 lines) ⚠️
├── command_runner.py          # Mixed concerns (412 lines) ⚠️
├── binary_runner.py           # Simple wrapper (67 lines)
├── cusfm_cli.py               # CLI parser (322 lines)
├── bin/                       # 22 binaries
├── lib/                       # Shared libraries
├── configs/                   # Protobuf configs
└── models/                    # ONNX models
```

### 2.2 Current Responsibilities

#### CuSFMRunner (1088 lines) - TOO MANY RESPONSIBILITIES
- ✗ Pipeline orchestration (21 methods)
- ✗ Directory creation and management
- ✗ File path resolution
- ✗ Configuration parsing
- ✗ Command building
- ✗ Error handling
- ✗ Logging configuration
- ✗ Multi-track coordination
- ✗ Stereo camera handling
- ✗ Format conversions

#### CommandRunner (412 lines) - MIXED CONCERNS
- ✗ Binary execution
- ✗ Environment setup (LD_LIBRARY_PATH, TensorRT)
- ✗ Directory validation
- ✗ Runtime performance logging
- ✗ Thread pool management
- ✗ Dry-run simulation

---

## 3. SOLID Principles Violations

### ❌ Single Responsibility Principle (SRP)

**Violation 1: CuSFMRunner**
```python
# Current: One class doing EVERYTHING
class CuSFMRunner:
    def run_all(self):              # Orchestration
    def run_cuvslam(self):          # SLAM execution
    def extract_features(self):     # Feature extraction
    def generate_bow(self):         # BoW generation
    def generate_associations(self): # Association
    def pose_graph(self):           # Pose graph optimization
    def match_features(self):       # Feature matching
    def map_keypoints(self):        # 3D mapping
    def convert_map(self):          # Format conversion
    # ... 12+ more methods ...
```

**Issue**: Changes to logging affect mapping, changes to directory structure affect feature extraction, etc.

**Violation 2: CommandRunner**
```python
class CommandRunner:
    def run_binary(self):                      # Execution
    def _setup_tensorrt_library_path(self):    # Environment
    def ensure_directory_exists(self):         # File system
    def _log_runtime(self):                    # Logging
```

**Issue**: File system concerns mixed with command execution.

### ❌ Open/Closed Principle (OCP)

**Violation: Hard-coded Pipeline Steps**
```python
def run_all(self):
    if not self.skip_cuvslam:
        self.run_cuvslam()
    if not self.skip_feature_extractor:
        self.extract_features()
    # ... hard-coded sequence ...
```

**Issue**: Adding a new pipeline step requires modifying `run_all()` method.

### ❌ Liskov Substitution Principle (LSP)

**Current State**: No inheritance used (N/A)

### ❌ Interface Segregation Principle (ISP)

**Violation: God Object Constructor**
```python
def __init__(self, input_dir, cusfm_base_dir, binary_dir, config_dir,
             model_dir, mask_dir, feature_type, skip_feature_extractor,
             skip_vocab_generator, skip_pose_graph, skip_matcher,
             skip_mapper, skip_map_convertor, skip_cuvslam, enable_debug,
             num_threads, override_frames_meta_file, cuvgl_dir,
             optimize_extrinsics, ba_frame_type, min_inter_frame_distance,
             min_inter_frame_rotation_degrees, dry_run, multi_track_input,
             # ... 20+ more parameters ...
):
```

**Issue**: 40+ constructor parameters. Clients forced to know about everything.

### ❌ Dependency Inversion Principle (DIP)

**Violation: Direct Dependencies on Concrete Classes**
```python
from .command_runner import CommandRunner  # Concrete class!

class CuSFMRunner:
    def __init__(self, ...):
        self.command_runner = CommandRunner(...)  # Direct instantiation
```

**Issue**: Cannot swap implementations for testing or different execution strategies.

---

## 4. Proposed Clean Architecture

### 4.1 Clean Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Frameworks & Drivers                      │
│  (External interfaces: CLI, Web API, File System, Binaries) │
│                                                              │
│  - cusfm_cli.py         (CLI entry point)                   │
│  - binary_executor.py   (Binary subprocess wrapper)         │
│  - file_repository.py   (File system I/O)                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   Interface Adapters                         │
│      (Controllers, Presenters, Gateways, Repositories)      │
│                                                              │
│  - pipeline_controller.py    (Orchestrates use cases)       │
│  - config_repository.py      (Loads configs)                │
│  - result_presenter.py       (Formats output)               │
│  - binary_gateway.py         (Binary execution interface)   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                      Use Cases                               │
│         (Application Business Rules - Pipeline Steps)       │
│                                                              │
│  - run_cuvslam_use_case.py          (Visual SLAM)           │
│  - extract_features_use_case.py     (Feature extraction)    │
│  - generate_bow_use_case.py         (BoW vocabulary)        │
│  - generate_associations_use_case.py (Associations)         │
│  - optimize_pose_graph_use_case.py  (Pose graph)            │
│  - match_features_use_case.py       (Feature matching)      │
│  - map_keypoints_use_case.py        (3D mapping)            │
│  - convert_to_colmap_use_case.py    (Format conversion)     │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                        Entities                              │
│          (Enterprise Business Rules - Domain Model)         │
│                                                              │
│  - sfm_project.py           (SfM project aggregate)         │
│  - camera_rig.py            (Camera rig entity)             │
│  - feature_collection.py    (Feature set value object)      │
│  - pose_graph.py            (Pose graph entity)             │
│  - keypoint_map.py          (3D map entity)                 │
│  - pipeline_config.py       (Configuration value object)    │
│  - frame_metadata.py        (Frame metadata entity)         │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Directory Structure (Proposed)

```
pycusfm/
├── __init__.py
│
├── domain/                          # LAYER 1: Entities
│   ├── __init__.py
│   ├── entities/
│   │   ├── __init__.py
│   │   ├── sfm_project.py          # Main aggregate root
│   │   ├── camera_rig.py           # Camera configuration
│   │   ├── frame_metadata.py       # Image frame data
│   │   ├── feature_collection.py   # Detected features
│   │   ├── pose_graph.py           # Camera poses
│   │   └── keypoint_map.py         # 3D reconstruction
│   ├── value_objects/
│   │   ├── __init__.py
│   │   ├── camera_intrinsics.py    # Camera parameters
│   │   ├── pose.py                 # 6DOF pose
│   │   ├── feature_descriptor.py   # Feature descriptors
│   │   └── match_pair.py           # Image pair matches
│   ├── enums/
│   │   ├── __init__.py
│   │   ├── feature_type.py         # SIFT, ALIKED, SuperPoint
│   │   ├── pipeline_mode.py        # Isaac, AV, Localization
│   │   └── coordinate_frame.py     # Camera, vehicle, rig frames
│   └── exceptions/
│       ├── __init__.py
│       ├── domain_exceptions.py    # Business rule violations
│       └── validation_errors.py    # Invalid data errors
│
├── application/                     # LAYER 2: Use Cases
│   ├── __init__.py
│   ├── interfaces/                 # Ports (abstractions)
│   │   ├── __init__.py
│   │   ├── binary_executor.py      # ABC for binary execution
│   │   ├── config_repository.py    # ABC for config loading
│   │   ├── file_repository.py      # ABC for file I/O
│   │   └── logger.py               # ABC for logging
│   ├── use_cases/
│   │   ├── __init__.py
│   │   ├── base_use_case.py        # Abstract base use case
│   │   ├── cuvslam/
│   │   │   ├── __init__.py
│   │   │   └── run_cuvslam.py      # Visual SLAM use case
│   │   ├── features/
│   │   │   ├── __init__.py
│   │   │   ├── extract_features.py # Feature extraction
│   │   │   └── match_features.py   # Feature matching
│   │   ├── bow/
│   │   │   ├── __init__.py
│   │   │   ├── generate_vocabulary.py
│   │   │   └── generate_associations.py
│   │   ├── pose_optimization/
│   │   │   ├── __init__.py
│   │   │   └── optimize_pose_graph.py
│   │   ├── mapping/
│   │   │   ├── __init__.py
│   │   │   ├── map_keypoints.py    # 3D mapping
│   │   │   └── bundle_adjustment.py
│   │   └── conversion/
│   │       ├── __init__.py
│   │       └── convert_to_colmap.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── pipeline_orchestrator.py # Coordinates use cases
│   │   ├── multi_track_coordinator.py
│   │   └── stereo_processor.py
│   └── dto/                        # Data Transfer Objects
│       ├── __init__.py
│       ├── pipeline_config_dto.py
│       ├── feature_extraction_request.py
│       └── mapping_result.py
│
├── infrastructure/                  # LAYER 3: Interface Adapters
│   ├── __init__.py
│   ├── binary/
│   │   ├── __init__.py
│   │   ├── binary_executor_impl.py # Concrete binary executor
│   │   ├── environment_manager.py  # LD_LIBRARY_PATH setup
│   │   └── performance_logger.py   # Runtime CSV logging
│   ├── config/
│   │   ├── __init__.py
│   │   ├── protobuf_config_loader.py
│   │   └── config_repository_impl.py
│   ├── persistence/
│   │   ├── __init__.py
│   │   ├── file_repository_impl.py # File I/O implementation
│   │   ├── json_serializer.py      # frames_meta.json handler
│   │   └── tum_pose_serializer.py  # TUM format handler
│   ├── logging/
│   │   ├── __init__.py
│   │   ├── logger_impl.py          # Concrete logger
│   │   └── log_formatter.py
│   └── external/
│       ├── __init__.py
│       └── tensorrt_manager.py     # TensorRT path management
│
├── presentation/                    # LAYER 4: Frameworks & Drivers
│   ├── __init__.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── cusfm_cli.py            # Main CLI entry
│   │   ├── argument_parser.py      # Argument parsing
│   │   ├── command_factory.py      # Creates use case instances
│   │   └── result_presenter.py     # Output formatting
│   └── api/                        # Future: REST API
│       └── __init__.py
│
├── shared/                          # Shared utilities
│   ├── __init__.py
│   ├── constants.py                # Directory/file constants
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── path_utils.py
│   │   └── validation_utils.py
│   └── types/
│       ├── __init__.py
│       └── common_types.py
│
├── bin/                            # Compiled binaries (unchanged)
├── lib/                            # Shared libraries (unchanged)
├── configs/                        # Protobuf configs (unchanged)
└── models/                         # ONNX models (unchanged)
```

### 4.3 Dependency Flow

```
CLI (presentation)
  ↓ depends on
Pipeline Orchestrator (application/services)
  ↓ depends on
Use Cases (application/use_cases)
  ↓ depends on
Entities (domain/entities)
  ↑ depend on
Interfaces (application/interfaces) ← implemented by → Infrastructure
```

**Key Principle**: Dependencies point INWARD. Domain has NO dependencies.

---

## 5. Detailed Refactoring Plan

### 5.1 Phase 1: Domain Layer (Entities)

#### 5.1.1 Create Core Entities

**File: `domain/entities/sfm_project.py`**
```python
"""SfM Project - Aggregate Root"""
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
from ..value_objects.pose import Pose
from ..enums.pipeline_mode import PipelineMode

@dataclass
class SfMProject:
    """
    Aggregate root for a Structure-from-Motion project.

    Represents the entire reconstruction project with all its components.
    """
    project_id: str
    input_directory: Path
    workspace_directory: Path
    mode: PipelineMode
    camera_rigs: List['CameraRig'] = field(default_factory=list)
    frames: List['FrameMetadata'] = field(default_factory=list)
    feature_collections: dict = field(default_factory=dict)
    pose_graph: Optional['PoseGraph'] = None
    keypoint_map: Optional['KeypointMap'] = None

    def add_camera_rig(self, rig: 'CameraRig') -> None:
        """Add a camera rig to the project."""
        self.camera_rigs.append(rig)

    def get_total_frames(self) -> int:
        """Get total number of frames across all cameras."""
        return len(self.frames)

    def validate(self) -> None:
        """Validate project consistency."""
        if not self.input_directory.exists():
            raise ValueError(f"Input directory does not exist: {self.input_directory}")
        if not self.camera_rigs:
            raise ValueError("At least one camera rig is required")
```

**File: `domain/entities/camera_rig.py`**
```python
"""Camera Rig Entity"""
from dataclasses import dataclass
from typing import List, Optional
from ..value_objects.camera_intrinsics import CameraIntrinsics
from ..value_objects.pose import Pose

@dataclass
class CameraRig:
    """
    Represents a camera or set of stereo cameras.

    Business rules:
    - Stereo pairs must have baseline > 0
    - All cameras in rig share synchronized timestamps
    """
    rig_id: str
    cameras: List['Camera']
    is_stereo: bool = False
    baseline: Optional[float] = None  # meters

    def validate_stereo_configuration(self) -> None:
        """Validate stereo camera setup."""
        if self.is_stereo:
            if len(self.cameras) != 2:
                raise ValueError("Stereo rig must have exactly 2 cameras")
            if self.baseline is None or self.baseline <= 0:
                raise ValueError("Stereo rig must have positive baseline")

@dataclass
class Camera:
    """Individual camera in a rig."""
    camera_id: str
    intrinsics: CameraIntrinsics
    extrinsics: Optional[Pose] = None  # Relative to rig
```

**File: `domain/value_objects/pose.py`**
```python
"""Pose Value Object"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)  # Immutable!
class Pose:
    """
    6DOF pose (position + orientation).

    Immutable value object - cannot be changed after creation.
    """
    position: tuple[float, float, float]  # (x, y, z)
    orientation: tuple[float, float, float, float]  # Quaternion (w, x, y, z)
    timestamp: int  # microseconds
    frame: str = "camera_frame"  # camera_frame, vehicle_frame, etc.

    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 transformation matrix."""
        # Implementation...
        pass

    def transform_to_frame(self, target_frame: str, transform: np.ndarray) -> 'Pose':
        """Transform pose to different coordinate frame."""
        # Returns new Pose (immutable)
        pass
```

#### 5.1.2 Create Enums

**File: `domain/enums/feature_type.py`**
```python
"""Feature Type Enumeration"""
from enum import Enum

class FeatureType(Enum):
    """Supported feature detector types."""
    SIFT = "sift"
    ALIKED = "aliked"
    SUPERPOINT = "superpoint"

    @property
    def requires_tensorrt(self) -> bool:
        """Check if feature type needs TensorRT."""
        return self in (FeatureType.ALIKED, FeatureType.SUPERPOINT)

    @property
    def model_path_suffix(self) -> str:
        """Get model directory name."""
        if self == FeatureType.ALIKED:
            return "aliked_lightglue"
        elif self == FeatureType.SUPERPOINT:
            return "superpoint_lightglue"
        return ""
```

**File: `domain/enums/pipeline_mode.py`**
```python
"""Pipeline Mode Enumeration"""
from enum import Enum

class PipelineMode(Enum):
    """Pipeline execution modes."""
    ISAAC_ROBOTICS = "isaac"      # Stereo cameras, synchronized
    AUTONOMOUS_VEHICLE = "av"     # Rolling video, flexible frames
    LOCALIZATION = "localization" # Match against existing map
    MULTI_TRACK = "multi_track"   # Multiple camera rigs/times
```

### 5.2 Phase 2: Application Layer (Use Cases)

#### 5.2.1 Define Interfaces (Dependency Inversion)

**File: `application/interfaces/binary_executor.py`**
```python
"""Binary Executor Interface (Port)"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ExecutionResult:
    """Result of binary execution."""
    exit_code: int
    stdout: str
    stderr: str
    execution_time_seconds: float
    success: bool

class IBinaryExecutor(ABC):
    """
    Interface for executing compiled binaries.

    Implementations can be: subprocess, Docker, remote execution, mock, etc.
    """

    @abstractmethod
    def execute(
        self,
        binary_name: str,
        arguments: List[str],
        env_vars: Dict[str, str] = None
    ) -> ExecutionResult:
        """
        Execute a binary with given arguments.

        Args:
            binary_name: Name of binary (e.g., "feature_extractor_main")
            arguments: List of command-line arguments
            env_vars: Optional environment variables

        Returns:
            ExecutionResult with stdout, stderr, exit code
        """
        pass

    @abstractmethod
    def execute_parallel(
        self,
        tasks: List[tuple[str, List[str]]],
        max_workers: int = 4
    ) -> List[ExecutionResult]:
        """Execute multiple binaries in parallel."""
        pass
```

**File: `application/interfaces/config_repository.py`**
```python
"""Configuration Repository Interface"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

class IConfigRepository(ABC):
    """Interface for loading configuration files."""

    @abstractmethod
    def load_protobuf_config(self, config_name: str) -> Dict[str, Any]:
        """Load a protobuf configuration file."""
        pass

    @abstractmethod
    def get_config_path(self, config_name: str) -> Path:
        """Get full path to a configuration file."""
        pass
```

**File: `application/interfaces/file_repository.py`**
```python
"""File Repository Interface"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

class IFileRepository(ABC):
    """Interface for file system operations."""

    @abstractmethod
    def read_json(self, file_path: Path) -> Dict[str, Any]:
        """Read JSON file."""
        pass

    @abstractmethod
    def write_json(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Write JSON file."""
        pass

    @abstractmethod
    def ensure_directory_exists(self, directory: Path) -> None:
        """Create directory if it doesn't exist."""
        pass

    @abstractmethod
    def list_images(self, directory: Path, extensions: List[str]) -> List[Path]:
        """List image files in directory."""
        pass
```

#### 5.2.2 Create Base Use Case

**File: `application/use_cases/base_use_case.py`**
```python
"""Base Use Case"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

# Generic types for request and response
TRequest = TypeVar('TRequest')
TResponse = TypeVar('TResponse')

@dataclass
class UseCaseResult(Generic[TResponse]):
    """Standard use case result wrapper."""
    success: bool
    data: TResponse = None
    error: str = None

class BaseUseCase(ABC, Generic[TRequest, TResponse]):
    """
    Abstract base class for all use cases.

    Each use case represents one step in the SfM pipeline.
    """

    @abstractmethod
    def execute(self, request: TRequest) -> UseCaseResult[TResponse]:
        """
        Execute the use case.

        Args:
            request: Input data for the use case

        Returns:
            UseCaseResult containing success status and data/error
        """
        pass

    def _validate_request(self, request: TRequest) -> None:
        """Override to add request validation."""
        pass
```

#### 5.2.3 Implement Feature Extraction Use Case

**File: `application/use_cases/features/extract_features.py`**
```python
"""Feature Extraction Use Case"""
from dataclasses import dataclass
from typing import List
from pathlib import Path
from ..base_use_case import BaseUseCase, UseCaseResult
from ...interfaces.binary_executor import IBinaryExecutor
from ...interfaces.file_repository import IFileRepository
from ...interfaces.logger import ILogger
from ....domain.entities.sfm_project import SfMProject
from ....domain.enums.feature_type import FeatureType

@dataclass
class FeatureExtractionRequest:
    """Input for feature extraction."""
    project: SfMProject
    feature_type: FeatureType
    max_keypoints: int = -1  # -1 = use config default
    batch_size: int = 0      # 0 = auto
    enable_debug: bool = False

@dataclass
class FeatureExtractionResponse:
    """Output from feature extraction."""
    features_directory: Path
    num_images_processed: int
    total_keypoints: int
    execution_time_seconds: float

class ExtractFeaturesUseCase(
    BaseUseCase[FeatureExtractionRequest, FeatureExtractionResponse]
):
    """
    Extract visual features from images.

    Responsibilities:
    - Build command arguments for feature_extractor_main
    - Execute binary via IBinaryExecutor
    - Parse results
    - Update project state
    """

    def __init__(
        self,
        binary_executor: IBinaryExecutor,
        file_repository: IFileRepository,
        logger: ILogger
    ):
        self.binary_executor = binary_executor
        self.file_repository = file_repository
        self.logger = logger

    def execute(
        self,
        request: FeatureExtractionRequest
    ) -> UseCaseResult[FeatureExtractionResponse]:
        """Execute feature extraction."""
        try:
            self._validate_request(request)

            # Prepare output directory
            features_dir = request.project.workspace_directory / "keyframes"
            self.file_repository.ensure_directory_exists(features_dir)

            # Build command arguments
            args = self._build_arguments(request, features_dir)

            # Execute binary
            self.logger.info(f"Extracting {request.feature_type.value} features...")
            result = self.binary_executor.execute(
                binary_name="feature_extractor_main",
                arguments=args
            )

            if not result.success:
                return UseCaseResult(
                    success=False,
                    error=f"Feature extraction failed: {result.stderr}"
                )

            # Parse results and create response
            response = FeatureExtractionResponse(
                features_directory=features_dir,
                num_images_processed=self._count_processed_images(features_dir),
                total_keypoints=0,  # Parse from output
                execution_time_seconds=result.execution_time_seconds
            )

            self.logger.info(
                f"Extracted features from {response.num_images_processed} images "
                f"in {response.execution_time_seconds:.2f}s"
            )

            return UseCaseResult(success=True, data=response)

        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
            return UseCaseResult(success=False, error=str(e))

    def _validate_request(self, request: FeatureExtractionRequest) -> None:
        """Validate the request."""
        if not request.project.input_directory.exists():
            raise ValueError(f"Input directory does not exist: {request.project.input_directory}")
        if request.max_keypoints < -1:
            raise ValueError("max_keypoints must be >= -1")

    def _build_arguments(
        self,
        request: FeatureExtractionRequest,
        output_dir: Path
    ) -> List[str]:
        """Build command-line arguments for feature_extractor_main."""
        args = [
            "--input_dir", str(request.project.input_directory),
            "--output_dir", str(output_dir),
            "--feature_type", request.feature_type.value,
        ]

        if request.max_keypoints > 0:
            args.extend(["--max_keypoints", str(request.max_keypoints)])

        if request.batch_size > 0:
            args.extend(["--batch_size", str(request.batch_size)])

        return args

    def _count_processed_images(self, features_dir: Path) -> int:
        """Count number of processed images."""
        # Implementation...
        return 0
```

### 5.3 Phase 3: Infrastructure Layer

#### 5.3.1 Binary Executor Implementation

**File: `infrastructure/binary/binary_executor_impl.py`**
```python
"""Concrete Binary Executor Implementation"""
import subprocess
import time
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from ...application.interfaces.binary_executor import (
    IBinaryExecutor, ExecutionResult
)
from .environment_manager import EnvironmentManager

class BinaryExecutorImpl(IBinaryExecutor):
    """
    Subprocess-based binary executor.

    Implements IBinaryExecutor using subprocess.run().
    """

    def __init__(
        self,
        binary_directory: Path,
        environment_manager: EnvironmentManager,
        dry_run: bool = False
    ):
        self.binary_directory = binary_directory
        self.env_manager = environment_manager
        self.dry_run = dry_run

    def execute(
        self,
        binary_name: str,
        arguments: List[str],
        env_vars: Dict[str, str] = None
    ) -> ExecutionResult:
        """Execute a binary."""
        binary_path = self.binary_directory / binary_name

        if not binary_path.exists():
            raise FileNotFoundError(f"Binary not found: {binary_path}")

        # Prepare environment
        env = self.env_manager.prepare_environment(env_vars)

        # Build command
        command = [str(binary_path)] + arguments

        if self.dry_run:
            print(f"DRY RUN: {' '.join(command)}")
            return ExecutionResult(
                exit_code=0,
                stdout="",
                stderr="",
                execution_time_seconds=0.0,
                success=True
            )

        # Execute
        start_time = time.time()
        result = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True
        )
        execution_time = time.time() - start_time

        return ExecutionResult(
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            execution_time_seconds=execution_time,
            success=result.returncode == 0
        )

    def execute_parallel(
        self,
        tasks: List[tuple[str, List[str]]],
        max_workers: int = 4
    ) -> List[ExecutionResult]:
        """Execute multiple binaries in parallel."""
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self.execute, binary, args): (binary, args)
                for binary, args in tasks
            }

            for future in as_completed(future_to_task):
                results.append(future.result())

        return results
```

**File: `infrastructure/binary/environment_manager.py`**
```python
"""Environment Variable Manager"""
import os
from typing import Dict, Optional
from pathlib import Path

class EnvironmentManager:
    """
    Manages environment variables for binary execution.

    Responsibilities:
    - Set up LD_LIBRARY_PATH for TensorRT
    - Configure CUDA paths
    - Manage packaged library paths
    """

    def __init__(
        self,
        lib_directory: Optional[Path] = None,
        add_tensorrt_path: bool = True
    ):
        self.lib_directory = lib_directory
        self.add_tensorrt_path = add_tensorrt_path

    def prepare_environment(
        self,
        additional_vars: Dict[str, str] = None
    ) -> Dict[str, str]:
        """Prepare environment variables for execution."""
        env = os.environ.copy()

        # Add packaged libraries
        if self.lib_directory and self.lib_directory.exists():
            self._add_library_path(env, self.lib_directory)

        # Add TensorRT paths
        if self.add_tensorrt_path:
            self._add_tensorrt_paths(env)

        # Add custom variables
        if additional_vars:
            env.update(additional_vars)

        return env

    def _add_library_path(self, env: Dict[str, str], lib_dir: Path) -> None:
        """Add library directory to LD_LIBRARY_PATH."""
        current_ld_path = env.get('LD_LIBRARY_PATH', '')
        if current_ld_path:
            env['LD_LIBRARY_PATH'] = f"{lib_dir}:{current_ld_path}"
        else:
            env['LD_LIBRARY_PATH'] = str(lib_dir)

    def _add_tensorrt_paths(self, env: Dict[str, str]) -> None:
        """Add TensorRT library paths."""
        # Implementation based on architecture
        import platform
        arch = platform.machine()
        if arch == 'x86_64':
            trt_path = '/usr/lib/x86_64-linux-gnu'
        elif arch == 'aarch64':
            trt_path = '/usr/lib/aarch64-linux-gnu'
        else:
            return

        self._add_library_path(env, Path(trt_path))
```

### 5.4 Phase 4: Presentation Layer (CLI)

**File: `presentation/cli/command_factory.py`**
```python
"""Command Factory - Dependency Injection Container"""
from pathlib import Path
from ...application.use_cases.features.extract_features import ExtractFeaturesUseCase
from ...application.use_cases.cuvslam.run_cuvslam import RunCuVSLAMUseCase
# ... other use cases
from ...infrastructure.binary.binary_executor_impl import BinaryExecutorImpl
from ...infrastructure.binary.environment_manager import EnvironmentManager
from ...infrastructure.persistence.file_repository_impl import FileRepositoryImpl
from ...infrastructure.config.config_repository_impl import ConfigRepositoryImpl
from ...infrastructure.logging.logger_impl import LoggerImpl

class CommandFactory:
    """
    Factory for creating use cases with dependencies.

    This is our Dependency Injection container.
    """

    def __init__(
        self,
        binary_dir: Path,
        config_dir: Path,
        lib_dir: Path,
        dry_run: bool = False
    ):
        # Create infrastructure components
        self.env_manager = EnvironmentManager(lib_directory=lib_dir)
        self.binary_executor = BinaryExecutorImpl(
            binary_directory=binary_dir,
            environment_manager=self.env_manager,
            dry_run=dry_run
        )
        self.file_repository = FileRepositoryImpl()
        self.config_repository = ConfigRepositoryImpl(config_dir)
        self.logger = LoggerImpl()

    def create_extract_features_use_case(self) -> ExtractFeaturesUseCase:
        """Create feature extraction use case."""
        return ExtractFeaturesUseCase(
            binary_executor=self.binary_executor,
            file_repository=self.file_repository,
            logger=self.logger
        )

    def create_run_cuvslam_use_case(self) -> RunCuVSLAMUseCase:
        """Create cuVSLAM use case."""
        return RunCuVSLAMUseCase(
            binary_executor=self.binary_executor,
            file_repository=self.file_repository,
            logger=self.logger
        )

    # ... factories for other use cases
```

**File: `presentation/cli/cusfm_cli.py`** (Refactored)
```python
"""CLI Entry Point - Clean Architecture Version"""
import argparse
from pathlib import Path
from .command_factory import CommandFactory
from .argument_parser import create_argument_parser
from ...application.services.pipeline_orchestrator import PipelineOrchestrator
from ...domain.entities.sfm_project import SfMProject
from ...domain.enums.pipeline_mode import PipelineMode
from ...domain.enums.feature_type import FeatureType

def main():
    """Main CLI entry point."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Create factory (DI container)
    factory = CommandFactory(
        binary_dir=Path(args.binary_dir) if args.binary_dir else get_default_binary_dir(),
        config_dir=Path(args.config_dir) if args.config_dir else get_default_config_dir(),
        lib_dir=get_default_lib_dir(),
        dry_run=args.dry_run
    )

    # Create domain entity (SfM Project)
    project = SfMProject(
        project_id=Path(args.cusfm_base_dir).name,
        input_directory=Path(args.input_dir) if args.input_dir else None,
        workspace_directory=Path(args.cusfm_base_dir),
        mode=_determine_pipeline_mode(args)
    )

    # Create orchestrator
    orchestrator = PipelineOrchestrator(factory)

    # Configure pipeline steps
    steps_config = _create_steps_config(args)

    # Run pipeline
    result = orchestrator.run_pipeline(project, steps_config)

    # Present results
    if result.success:
        print(f"✓ Pipeline completed successfully in {result.total_time_seconds:.2f}s")
        print(f"  Output: {project.workspace_directory}")
    else:
        print(f"✗ Pipeline failed: {result.error}")
        exit(1)

def _determine_pipeline_mode(args) -> PipelineMode:
    """Determine pipeline mode from arguments."""
    if args.av_data:
        return PipelineMode.AUTONOMOUS_VEHICLE
    elif args.multi_track_input:
        return PipelineMode.MULTI_TRACK
    else:
        return PipelineMode.ISAAC_ROBOTICS

def _create_steps_config(args) -> dict:
    """Create pipeline configuration from CLI arguments."""
    return {
        'skip_cuvslam': args.skip_cuvslam,
        'skip_feature_extractor': args.skip_feature_extractor,
        # ... other configuration
        'feature_type': FeatureType(args.feature_type) if args.feature_type else FeatureType.SIFT,
        'max_keypoints': args.max_keypoints_num,
    }

if __name__ == "__main__":
    main()
```

### 5.5 Phase 5: Testing

#### 5.5.1 Test Structure

```
tests/
├── __init__.py
├── unit/
│   ├── __init__.py
│   ├── domain/
│   │   ├── test_sfm_project.py
│   │   ├── test_camera_rig.py
│   │   └── test_pose.py
│   ├── application/
│   │   ├── test_extract_features_use_case.py
│   │   ├── test_run_cuvslam_use_case.py
│   │   └── test_pipeline_orchestrator.py
│   └── infrastructure/
│       ├── test_binary_executor_impl.py
│       └── test_environment_manager.py
├── integration/
│   ├── __init__.py
│   ├── test_feature_extraction_pipeline.py
│   └── test_full_pipeline.py
├── fixtures/
│   ├── __init__.py
│   ├── mock_binary_executor.py
│   ├── mock_file_repository.py
│   └── sample_data/
└── conftest.py  # pytest fixtures
```

#### 5.5.2 Example Unit Test (Domain)

**File: `tests/unit/domain/test_camera_rig.py`**
```python
"""Test Camera Rig Entity"""
import pytest
from pycusfm.domain.entities.camera_rig import CameraRig, Camera
from pycusfm.domain.value_objects.camera_intrinsics import CameraIntrinsics

class TestCameraRig:
    """Test camera rig business rules."""

    def test_stereo_rig_requires_two_cameras(self):
        """Stereo rig must have exactly 2 cameras."""
        camera1 = Camera(
            camera_id="left",
            intrinsics=CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
        )

        rig = CameraRig(
            rig_id="stereo1",
            cameras=[camera1],
            is_stereo=True,
            baseline=0.12
        )

        with pytest.raises(ValueError, match="must have exactly 2 cameras"):
            rig.validate_stereo_configuration()

    def test_stereo_rig_requires_positive_baseline(self):
        """Stereo rig must have baseline > 0."""
        camera1 = Camera(camera_id="left", intrinsics=...)
        camera2 = Camera(camera_id="right", intrinsics=...)

        rig = CameraRig(
            rig_id="stereo1",
            cameras=[camera1, camera2],
            is_stereo=True,
            baseline=0.0  # Invalid!
        )

        with pytest.raises(ValueError, match="positive baseline"):
            rig.validate_stereo_configuration()
```

#### 5.5.3 Example Unit Test (Use Case)

**File: `tests/unit/application/test_extract_features_use_case.py`**
```python
"""Test Feature Extraction Use Case"""
import pytest
from pathlib import Path
from pycusfm.application.use_cases.features.extract_features import (
    ExtractFeaturesUseCase, FeatureExtractionRequest
)
from pycusfm.domain.entities.sfm_project import SfMProject
from pycusfm.domain.enums.feature_type import FeatureType
from pycusfm.domain.enums.pipeline_mode import PipelineMode
from tests.fixtures.mock_binary_executor import MockBinaryExecutor
from tests.fixtures.mock_file_repository import MockFileRepository
from tests.fixtures.mock_logger import MockLogger

class TestExtractFeaturesUseCase:
    """Test feature extraction use case."""

    @pytest.fixture
    def mock_executor(self):
        """Create mock binary executor."""
        return MockBinaryExecutor()

    @pytest.fixture
    def mock_file_repo(self):
        """Create mock file repository."""
        return MockFileRepository()

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return MockLogger()

    @pytest.fixture
    def use_case(self, mock_executor, mock_file_repo, mock_logger):
        """Create use case with mocks."""
        return ExtractFeaturesUseCase(
            binary_executor=mock_executor,
            file_repository=mock_file_repo,
            logger=mock_logger
        )

    @pytest.fixture
    def project(self, tmp_path):
        """Create test project."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        return SfMProject(
            project_id="test",
            input_directory=input_dir,
            workspace_directory=workspace,
            mode=PipelineMode.ISAAC_ROBOTICS
        )

    def test_execute_success(self, use_case, project, mock_executor):
        """Test successful feature extraction."""
        # Arrange
        request = FeatureExtractionRequest(
            project=project,
            feature_type=FeatureType.SIFT,
            max_keypoints=1000
        )

        # Act
        result = use_case.execute(request)

        # Assert
        assert result.success
        assert result.data.num_images_processed >= 0
        assert mock_executor.execute_called
        assert "feature_extractor_main" in mock_executor.last_binary_name

    def test_execute_fails_if_input_dir_missing(self, use_case):
        """Test validation fails if input directory doesn't exist."""
        # Arrange
        project = SfMProject(
            project_id="test",
            input_directory=Path("/nonexistent"),
            workspace_directory=Path("/tmp/workspace"),
            mode=PipelineMode.ISAAC_ROBOTICS
        )
        request = FeatureExtractionRequest(
            project=project,
            feature_type=FeatureType.SIFT
        )

        # Act
        result = use_case.execute(request)

        # Assert
        assert not result.success
        assert "does not exist" in result.error
```

---

## 6. Migration Strategy

### 6.1 Migration Approach: Strangler Fig Pattern

Instead of rewriting everything at once, we'll gradually replace the old code:

1. **Create new architecture alongside old code**
2. **Route new features through new architecture**
3. **Gradually migrate old features**
4. **Remove old code when fully replaced**

### 6.2 Migration Phases

#### Phase 1: Foundation (Week 1-2)
- ✓ Create domain layer (entities, value objects, enums)
- ✓ Create application interfaces (ports)
- ✓ Set up test structure
- ✓ No breaking changes to existing code

#### Phase 2: Infrastructure (Week 2-3)
- ✓ Implement binary executor
- ✓ Implement file repository
- ✓ Implement config repository
- ✓ Implement logger
- ✓ Test with existing binaries

#### Phase 3: Use Cases (Week 3-5)
- ✓ Implement one use case at a time
- ✓ Start with simplest: feature extraction
- ✓ Add tests for each use case
- ✓ Gradually add: cuVSLAM, BoW, pose graph, matching, mapping

#### Phase 4: Orchestration (Week 5-6)
- ✓ Implement pipeline orchestrator
- ✓ Create command factory (DI container)
- ✓ Wire up all use cases

#### Phase 5: CLI Refactor (Week 6-7)
- ✓ Refactor CLI to use new architecture
- ✓ Keep backward compatibility with arguments
- ✓ Add deprecation warnings for old code paths

#### Phase 6: Deprecation (Week 7-8)
- ✓ Mark old code as deprecated
- ✓ Update documentation
- ✓ Provide migration guide

#### Phase 7: Removal (Week 8-9)
- ✓ Remove old `cusfm_runner.py` (1088 lines)
- ✓ Remove old `command_runner.py` (412 lines)
- ✓ Clean up unused code
- ✓ Final testing

### 6.3 Backward Compatibility

During migration, both old and new code coexist:

```python
# Old CLI (deprecated but functional)
from pycusfm.cusfm_runner import CuSFMRunner  # Still works

# New CLI (recommended)
from pycusfm.presentation.cli.cusfm_cli import main
```

We'll use environment variable to control which path to use:

```bash
# Use old code (default during migration)
export PYCUSFM_USE_LEGACY=true
cusfm_cli --input_dir ...

# Use new code
export PYCUSFM_USE_LEGACY=false
cusfm_cli --input_dir ...
```

### 6.4 Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Breaking existing workflows | Keep old code functional during migration |
| Performance regression | Benchmark each use case against old implementation |
| Increased complexity | Provide clear documentation and examples |
| Team learning curve | Code reviews, pair programming, training sessions |
| Incomplete migration | Set hard deadline for completing migration |

---

## 7. Testing Strategy

### 7.1 Test Coverage Goals

- **Domain Layer**: 95%+ coverage (pure business logic)
- **Application Layer**: 90%+ coverage (use cases)
- **Infrastructure Layer**: 80%+ coverage (integration points)
- **Overall**: 85%+ coverage

### 7.2 Test Types

#### Unit Tests
- Test individual classes/functions in isolation
- Use mocks for dependencies
- Fast execution (< 1 second per test)
- Run on every commit

#### Integration Tests
- Test interactions between layers
- Use real file system (temp directories)
- Mock only external binaries
- Run before merging to main

#### End-to-End Tests
- Test full pipeline with real binaries
- Use small sample datasets
- Validate output format (COLMAP)
- Run nightly or before releases

#### Performance Tests
- Benchmark each use case
- Ensure no regression vs. old implementation
- Track execution times in CI

### 7.3 Testing Tools

```python
# pyproject.toml
[project.optional-dependencies]
test = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-mock>=3.10.0",
    "hypothesis>=6.70.0",  # Property-based testing
]
```

### 7.4 CI/CD Pipeline

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -e ".[test]"

      - name: Run unit tests
        run: |
          pytest tests/unit/ -v --cov=pycusfm --cov-report=xml

      - name: Run integration tests
        run: |
          pytest tests/integration/ -v

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## 8. Timeline & Phases

### 8.1 Detailed Timeline (9 weeks)

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Domain Layer | Entities, value objects, enums, exceptions |
| 2 | Application Interfaces | All port interfaces, base use case |
| 3 | Infrastructure (Part 1) | Binary executor, environment manager |
| 4 | Infrastructure (Part 2) | File repo, config repo, logger |
| 5 | Use Cases (Part 1) | Feature extraction, cuVSLAM use cases |
| 6 | Use Cases (Part 2) | BoW, pose graph, matching use cases |
| 7 | Use Cases (Part 3) | Mapping, conversion use cases + orchestrator |
| 8 | CLI Refactor | New CLI, command factory, backward compat |
| 9 | Deprecation & Cleanup | Remove old code, final testing, documentation |

### 8.2 Success Criteria

- ✓ All 8 pipeline steps implemented as use cases
- ✓ 85%+ test coverage
- ✓ 100% functional parity with old implementation
- ✓ Performance within 5% of old implementation
- ✓ Zero breaking changes to CLI arguments
- ✓ Documentation updated
- ✓ Migration guide provided

---

## 9. Benefits Summary

### 9.1 SOLID Principles Adherence

| Principle | How Achieved |
|-----------|--------------|
| **Single Responsibility** | Each use case handles one pipeline step; separate classes for execution, file I/O, logging |
| **Open/Closed** | New pipeline steps added without modifying orchestrator; new feature types via enums |
| **Liskov Substitution** | All implementations of IBinaryExecutor are interchangeable |
| **Interface Segregation** | Small, focused interfaces (IBinaryExecutor, IFileRepository, etc.) |
| **Dependency Inversion** | Use cases depend on interfaces, not concrete implementations |

### 9.2 Clean Architecture Benefits

| Layer | Benefit |
|-------|---------|
| **Domain** | Pure business logic, no framework dependencies, highly testable |
| **Application** | Reusable use cases, can be called from CLI, API, or tests |
| **Infrastructure** | Swappable implementations (subprocess → Docker, local → remote) |
| **Presentation** | Multiple interfaces possible (CLI, REST API, GUI) |

### 9.3 Developer Experience Improvements

- **Easier Onboarding**: Clear structure, well-defined responsibilities
- **Faster Development**: Add features without breaking existing code
- **Better Testing**: Mock dependencies easily, fast unit tests
- **Clearer Debugging**: Errors isolated to specific layers/use cases
- **Confident Refactoring**: Tests ensure no regressions

---

## 10. Next Steps

1. **Review this plan** with team/stakeholders
2. **Approve architecture** and directory structure
3. **Create feature branch** for refactoring
4. **Start Phase 1** (Domain Layer)
5. **Set up CI/CD** for automated testing
6. **Begin migration** following strangler fig pattern

---

## Appendix A: Key Architectural Decisions

### ADR-001: Use Clean Architecture
**Decision**: Adopt Clean Architecture with 4 layers
**Rationale**: Separation of concerns, testability, maintainability
**Alternatives Considered**: MVC, Hexagonal Architecture

### ADR-002: Dependency Injection via Factory
**Decision**: Use CommandFactory for DI instead of framework
**Rationale**: Simple, no external dependencies, explicit
**Alternatives Considered**: python-inject, dependency-injector

### ADR-003: Interfaces via ABC
**Decision**: Use Python Abstract Base Classes for interfaces
**Rationale**: Built-in, type-safe, IDE support
**Alternatives Considered**: Protocols (PEP 544), duck typing

### ADR-004: Immutable Value Objects
**Decision**: Use frozen dataclasses for value objects
**Rationale**: Thread-safe, hashable, prevents bugs
**Alternatives Considered**: Regular classes, named tuples

### ADR-005: Strangler Fig Migration
**Decision**: Gradual migration with backward compatibility
**Rationale**: Lower risk, continuous delivery
**Alternatives Considered**: Big-bang rewrite

---

## Appendix B: References

- **Clean Architecture** by Robert C. Martin
- **Domain-Driven Design** by Eric Evans
- **SOLID Principles**: https://en.wikipedia.org/wiki/SOLID
- **Strangler Fig Pattern**: https://martinfowler.com/bliki/StranglerFigApplication.html
- **Python ABC**: https://docs.python.org/3/library/abc.html
