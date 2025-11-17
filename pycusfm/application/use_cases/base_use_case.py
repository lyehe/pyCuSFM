# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Base Use Case"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, Optional

# Generic types for request and response
TRequest = TypeVar('TRequest')
TResponse = TypeVar('TResponse')


@dataclass
class UseCaseResult(Generic[TResponse]):
    """
    Standard use case result wrapper.

    Encapsulates the result of a use case execution,
    including success status, data, and error information.

    Attributes:
        success: Whether the use case executed successfully
        data: Response data (if successful)
        error: Error message (if failed)
        metadata: Additional metadata about execution
    """

    success: bool
    data: Optional[TResponse] = None
    error: Optional[str] = None
    metadata: dict = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

    @classmethod
    def ok(cls, data: TResponse, **metadata) -> 'UseCaseResult[TResponse]':
        """
        Create a successful result.

        Args:
            data: Response data
            **metadata: Additional metadata

        Returns:
            UseCaseResult with success=True
        """
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata) -> 'UseCaseResult[TResponse]':
        """
        Create a failed result.

        Args:
            error: Error message
            **metadata: Additional metadata

        Returns:
            UseCaseResult with success=False
        """
        return cls(success=False, error=error, metadata=metadata)


class BaseUseCase(ABC, Generic[TRequest, TResponse]):
    """
    Abstract base class for all use cases.

    Each use case represents a single application operation (e.g., one step
    in the SfM pipeline). Use cases are:
    - Focused on a single task (Single Responsibility)
    - Independent and reusable
    - Testable in isolation
    - Framework-agnostic

    Example:
        class ExtractFeaturesUseCase(BaseUseCase[FeatureRequest, FeatureResponse]):
            def execute(self, request):
                # Implementation
                pass
    """

    @abstractmethod
    def execute(self, request: TRequest) -> UseCaseResult[TResponse]:
        """
        Execute the use case.

        Args:
            request: Input data for the use case

        Returns:
            UseCaseResult containing success status and data/error

        Raises:
            Should not raise exceptions - errors should be captured
            in UseCaseResult.error
        """
        pass

    def _validate_request(self, request: TRequest) -> None:
        """
        Validate request before execution.

        Override this method to add request validation logic.

        Args:
            request: Request to validate

        Raises:
            ValueError: If request is invalid
        """
        if request is None:
            raise ValueError("Request cannot be None")
