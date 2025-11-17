# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration Repository Interface (Port)"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class IConfigRepository(ABC):
    """
    Interface for loading configuration files.

    Abstracts away the details of how configs are loaded
    (protobuf, JSON, YAML, etc.)

    Dependency Inversion Principle: Use cases depend on this interface.
    """

    @abstractmethod
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a configuration file.

        Args:
            config_name: Name of config file (e.g., "keypoint_creation_config.pb.txt")

        Returns:
            Configuration as dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        pass

    @abstractmethod
    def get_config_path(self, config_name: str) -> Path:
        """
        Get full path to a configuration file.

        Args:
            config_name: Name of config file

        Returns:
            Full path to config file

        Raises:
            FileNotFoundError: If config doesn't exist
        """
        pass

    @abstractmethod
    def config_exists(self, config_name: str) -> bool:
        """
        Check if a configuration file exists.

        Args:
            config_name: Name of config file

        Returns:
            bool: True if config exists
        """
        pass

    @abstractmethod
    def list_configs(self) -> list[str]:
        """
        List all available configuration files.

        Returns:
            List of config file names
        """
        pass
