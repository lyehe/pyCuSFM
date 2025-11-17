# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""File Repository Interface (Port)"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class IFileRepository(ABC):
    """
    Interface for file system operations.

    This abstraction allows us to:
    - Mock file operations in tests
    - Swap implementations (local FS, cloud storage, etc.)
    - Add cross-cutting concerns (logging, validation)

    Dependency Inversion Principle: Use cases depend on this interface.
    """

    @abstractmethod
    def read_json(self, file_path: Path) -> Dict[str, Any]:
        """
        Read and parse a JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            Parsed JSON as dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid
        """
        pass

    @abstractmethod
    def write_json(
        self,
        file_path: Path,
        data: Dict[str, Any],
        indent: int = 2
    ) -> None:
        """
        Write data to JSON file.

        Args:
            file_path: Path to write to
            data: Dictionary to serialize
            indent: JSON indentation level

        Raises:
            IOError: If write fails
        """
        pass

    @abstractmethod
    def read_text(self, file_path: Path) -> str:
        """
        Read text file contents.

        Args:
            file_path: Path to text file

        Returns:
            File contents as string

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        pass

    @abstractmethod
    def write_text(self, file_path: Path, content: str) -> None:
        """
        Write text to file.

        Args:
            file_path: Path to write to
            content: Text content

        Raises:
            IOError: If write fails
        """
        pass

    @abstractmethod
    def ensure_directory_exists(self, directory: Path) -> None:
        """
        Create directory if it doesn't exist.

        Args:
            directory: Directory path to create

        Raises:
            IOError: If directory creation fails
        """
        pass

    @abstractmethod
    def directory_exists(self, directory: Path) -> bool:
        """
        Check if directory exists.

        Args:
            directory: Directory path to check

        Returns:
            bool: True if directory exists
        """
        pass

    @abstractmethod
    def file_exists(self, file_path: Path) -> bool:
        """
        Check if file exists.

        Args:
            file_path: File path to check

        Returns:
            bool: True if file exists
        """
        pass

    @abstractmethod
    def list_files(
        self,
        directory: Path,
        pattern: str = "*",
        recursive: bool = False
    ) -> List[Path]:
        """
        List files in directory matching pattern.

        Args:
            directory: Directory to search
            pattern: Glob pattern (e.g., "*.jpg")
            recursive: Whether to search recursively

        Returns:
            List of file paths

        Raises:
            FileNotFoundError: If directory doesn't exist
        """
        pass

    @abstractmethod
    def copy_file(self, source: Path, destination: Path) -> None:
        """
        Copy a file.

        Args:
            source: Source file path
            destination: Destination file path

        Raises:
            FileNotFoundError: If source doesn't exist
            IOError: If copy fails
        """
        pass

    @abstractmethod
    def delete_file(self, file_path: Path) -> None:
        """
        Delete a file.

        Args:
            file_path: File to delete

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        pass
