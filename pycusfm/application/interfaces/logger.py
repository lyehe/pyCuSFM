# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Logger Interface (Port)"""

from abc import ABC, abstractmethod
from typing import Any


class ILogger(ABC):
    """
    Interface for logging.

    Abstracts logging implementation to allow:
    - Swapping logging backends
    - Testing without actual logging
    - Adding structured logging

    Dependency Inversion Principle: Use cases depend on this interface.
    """

    @abstractmethod
    def debug(self, message: str, **kwargs: Any) -> None:
        """
        Log debug message.

        Args:
            message: Log message
            **kwargs: Additional structured data
        """
        pass

    @abstractmethod
    def info(self, message: str, **kwargs: Any) -> None:
        """
        Log info message.

        Args:
            message: Log message
            **kwargs: Additional structured data
        """
        pass

    @abstractmethod
    def warning(self, message: str, **kwargs: Any) -> None:
        """
        Log warning message.

        Args:
            message: Log message
            **kwargs: Additional structured data
        """
        pass

    @abstractmethod
    def error(self, message: str, **kwargs: Any) -> None:
        """
        Log error message.

        Args:
            message: Log message
            **kwargs: Additional structured data
        """
        pass

    @abstractmethod
    def exception(self, message: str, exc_info: Exception = None, **kwargs: Any) -> None:
        """
        Log exception with traceback.

        Args:
            message: Log message
            exc_info: Exception object
            **kwargs: Additional structured data
        """
        pass
