# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Keypoint Map Entity"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
import numpy as np


@dataclass
class MapPoint:
    """
    3D map point.

    Represents a 3D point in the reconstructed map with observations
    from multiple frames.
    """

    point_id: int
    position: Tuple[float, float, float]  # (x, y, z) in world frame
    color: Tuple[int, int, int] = (128, 128, 128)  # RGB
    observations: List[Tuple[str, int]] = field(
        default_factory=list)  # List of (frame_id, keypoint_id)
    error: float = 0.0  # Reprojection error

    @property
    def x(self) -> float:
        """Get x coordinate."""
        return self.position[0]

    @property
    def y(self) -> float:
        """Get y coordinate."""
        return self.position[1]

    @property
    def z(self) -> float:
        """Get z coordinate."""
        return self.position[2]

    @property
    def num_observations(self) -> int:
        """Get number of observations."""
        return len(self.observations)

    def add_observation(self, frame_id: str, keypoint_id: int) -> None:
        """
        Add an observation of this 3D point.

        Args:
            frame_id: Frame identifier
            keypoint_id: Keypoint identifier in that frame
        """
        self.observations.append((frame_id, keypoint_id))

    def get_observing_frames(self) -> Set[str]:
        """
        Get set of frames that observe this point.

        Returns:
            Set of frame IDs
        """
        return {frame_id for frame_id, _ in self.observations}

    def __repr__(self) -> str:
        """String representation."""
        return (f"MapPoint(id={self.point_id}, "
                f"pos=({self.x:.2f}, {self.y:.2f}, {self.z:.2f}), "
                f"obs={self.num_observations})")


@dataclass
class KeypointMap:
    """
    3D keypoint map entity.

    Represents the reconstructed 3D map with map points and their observations.
    This is the output of bundle adjustment and 3D reconstruction.

    Attributes:
        project_id: Identifier of the project this map belongs to
        points: Dictionary mapping point_id to MapPoint
        is_refined: Whether the map has been refined by bundle adjustment
    """

    project_id: str
    points: Dict[int, MapPoint] = field(default_factory=dict)
    is_refined: bool = False

    def add_point(self, point: MapPoint) -> None:
        """
        Add a 3D point to the map.

        Args:
            point: Map point to add

        Raises:
            ValueError: If point ID already exists
        """
        if point.point_id in self.points:
            raise ValueError(
                f"Point {point.point_id} already exists in map")
        self.points[point.point_id] = point

    def get_point(self, point_id: int) -> MapPoint:
        """
        Get a map point by ID.

        Args:
            point_id: Point identifier

        Returns:
            MapPoint: The map point

        Raises:
            KeyError: If point not found
        """
        if point_id not in self.points:
            raise KeyError(f"Point {point_id} not in map")
        return self.points[point_id]

    def remove_point(self, point_id: int) -> None:
        """
        Remove a point from the map.

        Args:
            point_id: Point identifier to remove
        """
        if point_id in self.points:
            del self.points[point_id]

    def has_point(self, point_id: int) -> bool:
        """Check if map contains a point."""
        return point_id in self.points

    @property
    def num_points(self) -> int:
        """Get number of points in map."""
        return len(self.points)

    @property
    def is_empty(self) -> bool:
        """Check if map is empty."""
        return len(self.points) == 0

    def get_points_observed_by_frame(self, frame_id: str) -> List[MapPoint]:
        """
        Get all points observed by a specific frame.

        Args:
            frame_id: Frame identifier

        Returns:
            List of map points observed by this frame
        """
        return [
            point for point in self.points.values()
            if frame_id in point.get_observing_frames()
        ]

    def get_total_observations(self) -> int:
        """Get total number of observations across all points."""
        return sum(point.num_observations for point in self.points.values())

    def get_mean_reprojection_error(self) -> float:
        """
        Get mean reprojection error across all points.

        Returns:
            float: Mean error, or 0.0 if no points
        """
        if self.is_empty:
            return 0.0
        return sum(point.error
                   for point in self.points.values()) / self.num_points

    def filter_by_observation_count(self, min_observations: int = 2) -> None:
        """
        Remove points with fewer than minimum observations.

        Args:
            min_observations: Minimum number of observations required
        """
        points_to_remove = [
            point_id for point_id, point in self.points.items()
            if point.num_observations < min_observations
        ]
        for point_id in points_to_remove:
            self.remove_point(point_id)

    def mark_as_refined(self) -> None:
        """Mark map as refined by bundle adjustment."""
        self.is_refined = True

    def __repr__(self) -> str:
        """String representation."""
        status = "refined" if self.is_refined else "unrefined"
        mean_error = self.get_mean_reprojection_error()
        return (f"KeypointMap(project={self.project_id}, "
                f"points={self.num_points}, {status}, "
                f"mean_error={mean_error:.4f})")
