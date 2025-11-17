# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""Pose Graph Entity"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
from ..value_objects.pose import Pose
from ..exceptions.domain_exceptions import PoseGraphOptimizationException


@dataclass
class PoseGraphEdge:
    """
    Edge in the pose graph.

    Represents a relative pose constraint between two frames.
    """

    frame1_id: str
    frame2_id: str
    relative_pose: Pose
    information_matrix: List[List[float]] = field(
        default_factory=lambda: [[1.0] * 6] * 6)  # 6x6 for 6DOF
    edge_type: str = "visual"  # visual, odometry, loop_closure

    def __repr__(self) -> str:
        """String representation."""
        return f"Edge({self.frame1_id} -> {self.frame2_id}, {self.edge_type})"


@dataclass
class PoseGraph:
    """
    Pose graph entity.

    Represents a graph of camera poses with relative constraints (edges).
    Used for pose graph optimization and loop closure detection.

    Attributes:
        project_id: Identifier of the project this pose graph belongs to
        nodes: Dictionary mapping frame_id to Pose
        edges: List of pose graph edges (constraints)
        is_optimized: Whether the pose graph has been optimized
    """

    project_id: str
    nodes: Dict[str, Pose] = field(default_factory=dict)
    edges: List[PoseGraphEdge] = field(default_factory=list)
    is_optimized: bool = False

    def add_node(self, frame_id: str, pose: Pose) -> None:
        """
        Add a node (camera pose) to the graph.

        Args:
            frame_id: Frame identifier
            pose: Camera pose
        """
        self.nodes[frame_id] = pose

    def add_edge(self, edge: PoseGraphEdge) -> None:
        """
        Add an edge (constraint) to the graph.

        Args:
            edge: Pose graph edge

        Raises:
            ValueError: If referenced frames don't exist in graph
        """
        if edge.frame1_id not in self.nodes:
            raise ValueError(
                f"Frame {edge.frame1_id} not in pose graph nodes")
        if edge.frame2_id not in self.nodes:
            raise ValueError(
                f"Frame {edge.frame2_id} not in pose graph nodes")
        self.edges.append(edge)

    def get_pose(self, frame_id: str) -> Pose:
        """
        Get pose for a specific frame.

        Args:
            frame_id: Frame identifier

        Returns:
            Pose: Camera pose

        Raises:
            KeyError: If frame not in graph
        """
        if frame_id not in self.nodes:
            raise KeyError(f"Frame {frame_id} not in pose graph")
        return self.nodes[frame_id]

    def has_node(self, frame_id: str) -> bool:
        """Check if graph contains a node."""
        return frame_id in self.nodes

    @property
    def num_nodes(self) -> int:
        """Get number of nodes in graph."""
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        """Get number of edges in graph."""
        return len(self.edges)

    def get_connected_frames(self, frame_id: str) -> Set[str]:
        """
        Get all frames connected to a given frame.

        Args:
            frame_id: Frame identifier

        Returns:
            Set of connected frame IDs
        """
        connected = set()
        for edge in self.edges:
            if edge.frame1_id == frame_id:
                connected.add(edge.frame2_id)
            elif edge.frame2_id == frame_id:
                connected.add(edge.frame1_id)
        return connected

    def get_edges_for_frame(self,
                            frame_id: str) -> List[PoseGraphEdge]:
        """
        Get all edges connected to a frame.

        Args:
            frame_id: Frame identifier

        Returns:
            List of edges involving this frame
        """
        return [
            edge for edge in self.edges
            if edge.frame1_id == frame_id or edge.frame2_id == frame_id
        ]

    def mark_as_optimized(self) -> None:
        """Mark pose graph as optimized."""
        self.is_optimized = True

    def validate(self) -> None:
        """
        Validate pose graph structure.

        Raises:
            PoseGraphOptimizationException: If graph structure is invalid
        """
        if self.num_nodes == 0:
            raise PoseGraphOptimizationException(
                "Pose graph must have at least one node")

        # Check that all edge endpoints exist in nodes
        for edge in self.edges:
            if edge.frame1_id not in self.nodes:
                raise PoseGraphOptimizationException(
                    f"Edge references non-existent node: {edge.frame1_id}")
            if edge.frame2_id not in self.nodes:
                raise PoseGraphOptimizationException(
                    f"Edge references non-existent node: {edge.frame2_id}")

    def __repr__(self) -> str:
        """String representation."""
        status = "optimized" if self.is_optimized else "unoptimized"
        return (f"PoseGraph(project={self.project_id}, "
                f"nodes={self.num_nodes}, edges={self.num_edges}, {status})")
