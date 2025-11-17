# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Domain Exceptions"""


class DomainException(Exception):
    """
    Base exception for all domain-level errors.

    Domain exceptions represent business rule violations
    that occur in the domain layer.
    """

    def __init__(self, message: str, details: dict = None):
        """
        Initialize domain exception.

        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """String representation."""
        if self.details:
            details_str = ", ".join(
                f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class InvalidPoseException(DomainException):
    """Raised when a pose is invalid."""
    pass


class InvalidCameraConfigurationException(DomainException):
    """Raised when camera configuration is invalid."""
    pass


class InsufficientFeaturesException(DomainException):
    """Raised when there are insufficient features for processing."""
    pass


class PoseGraphOptimizationException(DomainException):
    """Raised when pose graph optimization fails."""
    pass


class BundleAdjustmentException(DomainException):
    """Raised when bundle adjustment fails."""
    pass


class IncompatibleFramesException(DomainException):
    """Raised when coordinate frames are incompatible."""
    pass
