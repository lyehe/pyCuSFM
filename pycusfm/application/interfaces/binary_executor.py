# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Binary Executor Interface (Port)"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """
    Result of binary execution.

    Attributes:
        exit_code: Process exit code
        stdout: Standard output from the process
        stderr: Standard error from the process
        execution_time_seconds: Time taken to execute
        success: Whether execution was successful
        command: The command that was executed (for debugging)
    """

    exit_code: int
    stdout: str
    stderr: str
    execution_time_seconds: float
    success: bool
    command: Optional[str] = None

    def __post_init__(self):
        """Set success based on exit code if not explicitly set."""
        if self.success is None:
            self.success = self.exit_code == 0


class IBinaryExecutor(ABC):
    """
    Interface for executing compiled binaries.

    This is a port (abstraction) that can be implemented by:
    - Subprocess executor (local execution)
    - Docker executor (containerized execution)
    - Remote executor (distributed execution)
    - Mock executor (testing)

    Dependency Inversion Principle: Use cases depend on this interface,
    not on concrete implementations.
    """

    @abstractmethod
    def execute(
        self,
        binary_name: str,
        arguments: List[str],
        env_vars: Optional[Dict[str, str]] = None,
        working_dir: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute a binary with given arguments.

        Args:
            binary_name: Name of binary (e.g., "feature_extractor_main")
            arguments: List of command-line arguments
            env_vars: Optional environment variables to set
            working_dir: Optional working directory

        Returns:
            ExecutionResult with stdout, stderr, exit code, etc.

        Raises:
            FileNotFoundError: If binary doesn't exist
            RuntimeError: If execution fails critically
        """
        pass

    @abstractmethod
    def execute_parallel(
        self,
        tasks: List[Tuple[str, List[str]]],
        max_workers: int = 4,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> List[ExecutionResult]:
        """
        Execute multiple binaries in parallel.

        Args:
            tasks: List of (binary_name, arguments) tuples
            max_workers: Maximum number of parallel workers
            env_vars: Optional environment variables for all tasks

        Returns:
            List of ExecutionResult objects (one per task)

        Raises:
            RuntimeError: If parallel execution setup fails
        """
        pass

    @abstractmethod
    def is_binary_available(self, binary_name: str) -> bool:
        """
        Check if a binary is available for execution.

        Args:
            binary_name: Name of binary to check

        Returns:
            bool: True if binary exists and is executable
        """
        pass
