# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Mock Logger for Testing"""

from typing import Any, List
from pycusfm.application.interfaces.logger import ILogger


class MockLogger(ILogger):
    """
    Mock implementation of ILogger for testing.

    Captures log messages for verification in tests.
    """

    def __init__(self):
        """Initialize mock logger."""
        self.debug_messages: List[str] = []
        self.info_messages: List[str] = []
        self.warning_messages: List[str] = []
        self.error_messages: List[str] = []
        self.exception_messages: List[str] = []

    def debug(self, message: str, **kwargs: Any) -> None:
        """Mock debug logging."""
        self.debug_messages.append(message)

    def info(self, message: str, **kwargs: Any) -> None:
        """Mock info logging."""
        self.info_messages.append(message)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Mock warning logging."""
        self.warning_messages.append(message)

    def error(self, message: str, **kwargs: Any) -> None:
        """Mock error logging."""
        self.error_messages.append(message)

    def exception(
        self, message: str, exc_info: Exception = None, **kwargs: Any
    ) -> None:
        """Mock exception logging."""
        self.exception_messages.append(message)

    def reset(self):
        """Reset mock state."""
        self.debug_messages.clear()
        self.info_messages.clear()
        self.warning_messages.clear()
        self.error_messages.clear()
        self.exception_messages.clear()
