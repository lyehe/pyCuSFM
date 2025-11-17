# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Mock Binary Executor for Testing"""

from typing import List, Dict, Optional, Tuple
from pycusfm.application.interfaces.binary_executor import (
    IBinaryExecutor,
    ExecutionResult,
)


class MockBinaryExecutor(IBinaryExecutor):
    """
    Mock implementation of IBinaryExecutor for testing.

    This demonstrates the power of Dependency Inversion Principle:
    tests can inject this mock instead of the real executor.
    """

    def __init__(self, should_succeed: bool = True):
        """
        Initialize mock executor.

        Args:
            should_succeed: Whether executions should succeed
        """
        self.should_succeed = should_succeed
        self.execution_history: List[Tuple[str, List[str]]] = []
        self.last_binary_name: Optional[str] = None
        self.last_arguments: Optional[List[str]] = None
        self.execute_called = False

    def execute(
        self,
        binary_name: str,
        arguments: List[str],
        env_vars: Optional[Dict[str, str]] = None,
        working_dir: Optional[str] = None,
    ) -> ExecutionResult:
        """Mock execute."""
        self.execute_called = True
        self.last_binary_name = binary_name
        self.last_arguments = arguments
        self.execution_history.append((binary_name, arguments))

        if self.should_succeed:
            return ExecutionResult(
                exit_code=0,
                stdout=f"Successfully executed {binary_name}",
                stderr="",
                execution_time_seconds=0.1,
                success=True,
                command=f"{binary_name} {' '.join(arguments)}",
            )
        else:
            return ExecutionResult(
                exit_code=1,
                stdout="",
                stderr=f"Failed to execute {binary_name}",
                execution_time_seconds=0.1,
                success=False,
                command=f"{binary_name} {' '.join(arguments)}",
            )

    def execute_parallel(
        self,
        tasks: List[Tuple[str, List[str]]],
        max_workers: int = 4,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> List[ExecutionResult]:
        """Mock execute parallel."""
        results = []
        for binary_name, arguments in tasks:
            result = self.execute(binary_name, arguments, env_vars)
            results.append(result)
        return results

    def is_binary_available(self, binary_name: str) -> bool:
        """Mock binary availability check."""
        return True

    def reset(self):
        """Reset mock state."""
        self.execute_called = False
        self.last_binary_name = None
        self.last_arguments = None
        self.execution_history.clear()
