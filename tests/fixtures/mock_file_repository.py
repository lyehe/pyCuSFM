# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Mock File Repository for Testing"""

from pathlib import Path
from typing import Any, Dict, List
from pycusfm.application.interfaces.file_repository import IFileRepository


class MockFileRepository(IFileRepository):
    """
    Mock implementation of IFileRepository for testing.

    Stores files in memory instead of disk.
    """

    def __init__(self):
        """Initialize mock repository."""
        self.files: Dict[Path, str] = {}
        self.directories: set[Path] = set()

    def read_json(self, file_path: Path) -> Dict[str, Any]:
        """Mock read JSON."""
        import json

        if file_path not in self.files:
            raise FileNotFoundError(f"File not found: {file_path}")
        return json.loads(self.files[file_path])

    def write_json(
        self, file_path: Path, data: Dict[str, Any], indent: int = 2
    ) -> None:
        """Mock write JSON."""
        import json

        self.files[file_path] = json.dumps(data, indent=indent)

    def read_text(self, file_path: Path) -> str:
        """Mock read text."""
        if file_path not in self.files:
            raise FileNotFoundError(f"File not found: {file_path}")
        return self.files[file_path]

    def write_text(self, file_path: Path, content: str) -> None:
        """Mock write text."""
        self.files[file_path] = content

    def ensure_directory_exists(self, directory: Path) -> None:
        """Mock directory creation."""
        self.directories.add(directory)

    def directory_exists(self, directory: Path) -> bool:
        """Mock directory exists check."""
        return directory in self.directories

    def file_exists(self, file_path: Path) -> bool:
        """Mock file exists check."""
        return file_path in self.files

    def list_files(
        self, directory: Path, pattern: str = "*", recursive: bool = False
    ) -> List[Path]:
        """Mock list files."""
        # Simple implementation: return all files in directory
        return [
            path for path in self.files.keys()
            if path.parent == directory
        ]

    def copy_file(self, source: Path, destination: Path) -> None:
        """Mock copy file."""
        if source not in self.files:
            raise FileNotFoundError(f"Source file not found: {source}")
        self.files[destination] = self.files[source]

    def delete_file(self, file_path: Path) -> None:
        """Mock delete file."""
        if file_path not in self.files:
            raise FileNotFoundError(f"File not found: {file_path}")
        del self.files[file_path]

    def reset(self):
        """Reset mock state."""
        self.files.clear()
        self.directories.clear()
