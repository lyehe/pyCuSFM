# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration and shared fixtures"""

import pytest
from pathlib import Path
import tempfile
import numpy as np


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image_path(temp_dir):
    """Create a sample image path."""
    image_path = temp_dir / "test_image.jpg"
    image_path.touch()
    return image_path


@pytest.fixture
def normalized_quaternion():
    """Return a normalized quaternion."""
    return (1.0, 0.0, 0.0, 0.0)  # Identity quaternion


@pytest.fixture
def sample_position():
    """Return a sample 3D position."""
    return (1.0, 2.0, 3.0)
