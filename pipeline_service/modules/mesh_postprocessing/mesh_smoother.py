"""
Mesh post-processing and smoothing service.

Provides various mesh smoothing algorithms to improve 3D model quality:
- Laplacian smoothing
- Taubin smoothing (prevents shrinkage)
- Normal refinement
"""

from __future__ import annotations

import io
import time
from typing import Optional, Tuple
import numpy as np
from PIL import Image

from config import Settings
from logger_config import logger


class MeshSmoothingService:
    """Service for mesh post-processing and smoothing operations."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.enable_smoothing = settings.enable_mesh_smoothing
        self.iterations = settings.smoothing_iterations
        self.strength = settings.smoothing_strength
        self.enable_normal_refinement = settings.enable_normal_refinement

        # Taubin smoothing parameters (prevents shrinkage)
        self.lambda_smooth = 0.5  # Smoothing factor
        self.mu_smooth = -0.53  # Inflation factor (slightly negative)

    async def startup(self) -> None:
        """Initialize the mesh smoothing service."""
        logger.info("Mesh smoothing service initialized")
        if not self.enable_smoothing:
            logger.info("Mesh smoothing is disabled")
        else:
            logger.info(
                f"Mesh smoothing enabled: iterations={self.iterations}, "
                f"strength={self.strength}, normal_refinement={self.enable_normal_refinement}"
            )

    async def shutdown(self) -> None:
        """Shutdown the mesh smoothing service."""
        logger.info("Mesh smoothing service closed")

    def smooth_ply(self, ply_bytes: bytes) -> bytes:
        """
        Apply smoothing to PLY mesh data.

        Args:
            ply_bytes: Input PLY file as bytes

        Returns:
            Smoothed PLY file as bytes
        """
        if not self.enable_smoothing:
            return ply_bytes

        try:
            t1 = time.time()

            # Parse PLY file
            vertices, faces, colors, normals, has_colors, has_normals = self._parse_ply(
                ply_bytes
            )

            if vertices is None or len(vertices) == 0:
                logger.warning("No vertices found in PLY, skipping smoothing")
                return ply_bytes

            # Apply smoothing
            smoothed_vertices = self._apply_taubin_smoothing(
                vertices, faces, self.iterations, self.strength
            )

            # Refine normals if enabled
            if self.enable_normal_refinement and faces is not None:
                normals = self._compute_vertex_normals(smoothed_vertices, faces)
                has_normals = True

            # Rebuild PLY
            smoothed_ply = self._build_ply(
                smoothed_vertices, faces, colors, normals, has_colors, has_normals
            )

            smoothing_time = time.time() - t1
            logger.success(
                f"Mesh smoothing complete in {smoothing_time:.2f}s - "
                f"Vertices: {len(vertices)}, Faces: {len(faces) if faces is not None else 0}"
            )

            return smoothed_ply

        except Exception as e:
            logger.error(f"Error during mesh smoothing: {e}")
            logger.warning("Returning original mesh without smoothing")
            return ply_bytes

    def _parse_ply(self, ply_bytes: bytes) -> Tuple[Optional[np.ndarray], ...]:
        """
        Parse PLY file format.

        Returns:
            Tuple of (vertices, faces, colors, normals, has_colors, has_normals)
        """
        try:
            lines = ply_bytes.decode("utf-8", errors="ignore").split("\n")
        except:
            lines = ply_bytes.decode("latin-1").split("\n")

        # Parse header
        vertex_count = 0
        face_count = 0
        has_colors = False
        has_normals = False
        properties = []
        in_header = True
        header_end_idx = 0

        for idx, line in enumerate(lines):
            line = line.strip()

            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("element face"):
                face_count = int(line.split()[-1])
            elif line.startswith("property"):
                properties.append(line)
                if "red" in line or "green" in line or "blue" in line:
                    has_colors = True
                if "nx" in line or "ny" in line or "nz" in line:
                    has_normals = True
            elif line == "end_header":
                in_header = False
                header_end_idx = idx + 1
                break

        if vertex_count == 0:
            return None, None, None, None, False, False

        # Parse vertex data
        vertices = []
        colors = [] if has_colors else None
        normals = [] if has_normals else None

        for i in range(header_end_idx, header_end_idx + vertex_count):
            if i >= len(lines):
                break
            parts = lines[i].strip().split()
            if len(parts) < 3:
                continue

            # Parse x, y, z
            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])

            # Parse normals if present (usually nx, ny, nz after x, y, z)
            if has_normals and len(parts) >= 6:
                normals.append([float(parts[3]), float(parts[4]), float(parts[5])])

            # Parse colors if present
            if has_colors:
                color_offset = 6 if has_normals else 3
                if len(parts) >= color_offset + 3:
                    colors.append(
                        [
                            int(parts[color_offset]),
                            int(parts[color_offset + 1]),
                            int(parts[color_offset + 2]),
                        ]
                    )

        vertices = np.array(vertices, dtype=np.float32)

        # Parse face data
        faces = []
        face_start_idx = header_end_idx + vertex_count

        for i in range(face_start_idx, face_start_idx + face_count):
            if i >= len(lines):
                break
            parts = lines[i].strip().split()
            if len(parts) < 4:
                continue

            # First number is vertex count, then vertex indices
            num_vertices = int(parts[0])
            if num_vertices == 3 and len(parts) >= 4:
                faces.append([int(parts[1]), int(parts[2]), int(parts[3])])

        faces = np.array(faces, dtype=np.int32) if faces else None
        colors = np.array(colors, dtype=np.uint8) if colors else None
        normals = (
            np.array(normals, dtype=np.float32)
            if normals and len(normals) > 0
            else None
        )

        return vertices, faces, colors, normals, has_colors, has_normals

    def _apply_taubin_smoothing(
        self,
        vertices: np.ndarray,
        faces: Optional[np.ndarray],
        iterations: int,
        strength: float,
    ) -> np.ndarray:
        """
        Apply Taubin smoothing (prevents mesh shrinkage).

        Taubin smoothing alternates between Laplacian smoothing and inflation
        to preserve volume while smoothing the surface.
        """
        if faces is None or len(faces) == 0:
            # No faces, apply simple vertex smoothing
            return self._apply_simple_smoothing(vertices, iterations, strength)

        # Build adjacency information
        adjacency = self._build_adjacency(vertices, faces)

        smoothed = vertices.copy()
        lambda_factor = self.lambda_smooth * strength
        mu_factor = self.mu_smooth * strength

        for iteration in range(iterations):
            # Smoothing step (lambda)
            smoothed = self._laplacian_smooth_step(smoothed, adjacency, lambda_factor)

            # Inflation step (mu) - prevents shrinkage
            smoothed = self._laplacian_smooth_step(smoothed, adjacency, mu_factor)

        return smoothed

    def _build_adjacency(self, vertices: np.ndarray, faces: np.ndarray) -> dict:
        """Build vertex adjacency information from faces."""
        adjacency = {i: set() for i in range(len(vertices))}

        for face in faces:
            # Add all edges of the triangle
            adjacency[face[0]].update([face[1], face[2]])
            adjacency[face[1]].update([face[0], face[2]])
            adjacency[face[2]].update([face[0], face[1]])

        return adjacency

    def _laplacian_smooth_step(
        self, vertices: np.ndarray, adjacency: dict, factor: float
    ) -> np.ndarray:
        """Single step of Laplacian smoothing."""
        smoothed = vertices.copy()

        for i, neighbors in adjacency.items():
            if not neighbors:
                continue

            # Compute average position of neighbors
            neighbor_positions = vertices[list(neighbors)]
            avg_position = np.mean(neighbor_positions, axis=0)

            # Move vertex toward average
            smoothed[i] = vertices[i] + factor * (avg_position - vertices[i])

        return smoothed

    def _apply_simple_smoothing(
        self, vertices: np.ndarray, iterations: int, strength: float
    ) -> np.ndarray:
        """Apply simple smoothing for point clouds without faces."""
        from scipy.spatial import cKDTree

        smoothed = vertices.copy()
        k_neighbors = min(10, len(vertices) - 1)

        if k_neighbors <= 0:
            return vertices

        for _ in range(iterations):
            tree = cKDTree(smoothed)

            for i in range(len(smoothed)):
                # Find k nearest neighbors
                distances, indices = tree.query(smoothed[i], k=k_neighbors + 1)
                neighbor_indices = indices[1:]  # Exclude self

                # Average position of neighbors
                avg_position = np.mean(smoothed[neighbor_indices], axis=0)

                # Move toward average
                smoothed[i] = smoothed[i] + strength * (avg_position - smoothed[i])

        return smoothed

    def _compute_vertex_normals(
        self, vertices: np.ndarray, faces: np.ndarray
    ) -> np.ndarray:
        """Compute smooth vertex normals from face normals."""
        normals = np.zeros_like(vertices)

        # Accumulate face normals
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]

            # Compute face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)

            # Accumulate to vertex normals
            normals[face[0]] += face_normal
            normals[face[1]] += face_normal
            normals[face[2]] += face_normal

        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Prevent division by zero
        normals = normals / norms

        return normals

    def _build_ply(
        self,
        vertices: np.ndarray,
        faces: Optional[np.ndarray],
        colors: Optional[np.ndarray],
        normals: Optional[np.ndarray],
        has_colors: bool,
        has_normals: bool,
    ) -> bytes:
        """Build PLY file from components."""
        lines = ["ply", "format ascii 1.0"]

        # Vertex element
        lines.append(f"element vertex {len(vertices)}")
        lines.append("property float x")
        lines.append("property float y")
        lines.append("property float z")

        if has_normals and normals is not None:
            lines.append("property float nx")
            lines.append("property float ny")
            lines.append("property float nz")

        if has_colors and colors is not None:
            lines.append("property uchar red")
            lines.append("property uchar green")
            lines.append("property uchar blue")

        # Face element
        if faces is not None and len(faces) > 0:
            lines.append(f"element face {len(faces)}")
            lines.append("property list uchar int vertex_indices")

        lines.append("end_header")

        # Write vertices
        for i, vertex in enumerate(vertices):
            parts = [f"{vertex[0]:.6f}", f"{vertex[1]:.6f}", f"{vertex[2]:.6f}"]

            if has_normals and normals is not None and i < len(normals):
                parts.extend(
                    [
                        f"{normals[i][0]:.6f}",
                        f"{normals[i][1]:.6f}",
                        f"{normals[i][2]:.6f}",
                    ]
                )

            if has_colors and colors is not None and i < len(colors):
                parts.extend([str(colors[i][0]), str(colors[i][1]), str(colors[i][2])])

            lines.append(" ".join(parts))

        # Write faces
        if faces is not None:
            for face in faces:
                lines.append(f"3 {face[0]} {face[1]} {face[2]}")

        ply_string = "\n".join(lines)
        return ply_string.encode("utf-8")
