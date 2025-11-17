# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Validation Errors"""

from typing import List, Dict, Any


class ValidationError(Exception):
    """
    Exception raised when entity or value object validation fails.

    This represents data validation errors, not business rule violations.
    """

    def __init__(self, message: str, errors: List[str] = None):
        """
        Initialize validation error.

        Args:
            message: Human-readable error message
            errors: List of specific validation errors
        """
        super().__init__(message)
        self.message = message
        self.errors = errors or []

    def add_error(self, error: str) -> None:
        """
        Add a validation error.

        Args:
            error: Error message to add
        """
        self.errors.append(error)

    def __str__(self) -> str:
        """String representation."""
        if self.errors:
            errors_str = "; ".join(self.errors)
            return f"{self.message}: {errors_str}"
        return self.message


class EntityValidationError(ValidationError):
    """Raised when entity validation fails."""

    def __init__(self,
                 entity_type: str,
                 entity_id: str,
                 errors: List[str] = None):
        """
        Initialize entity validation error.

        Args:
            entity_type: Type of entity that failed validation
            entity_id: Identifier of the entity
            errors: List of validation errors
        """
        message = f"{entity_type} validation failed for ID '{entity_id}'"
        super().__init__(message, errors)
        self.entity_type = entity_type
        self.entity_id = entity_id


class ValueObjectValidationError(ValidationError):
    """Raised when value object validation fails."""

    def __init__(self, value_object_type: str, errors: List[str] = None):
        """
        Initialize value object validation error.

        Args:
            value_object_type: Type of value object that failed validation
            errors: List of validation errors
        """
        message = f"{value_object_type} validation failed"
        super().__init__(message, errors)
        self.value_object_type = value_object_type
