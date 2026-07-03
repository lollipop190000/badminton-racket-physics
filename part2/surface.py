"""Relaxation-based string-lattice solver for Part 2 (Numba njit).
Provides surface(), make_rect_edge(), make_ellipse_edge() imported by
04_relaxation_force_map.ipynb. Extracted from the ellipse-optimization code."""

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# ------------------------------
# Boundary Generators
# ------------------------------
def make_rect_edge(nx, ny):
    edges = np.zeros((ny, nx), dtype=np.bool_)
    edges[0, :] = True
    edges[-1, :] = True
    edges[:, 0] = True
    edges[:, -1] = True
    return edges

def make_ellipse_edge(nx, ny, aspect_ratio=1.0):
    """
    aspect_ratio: major-axis / minor-axis ratio (>= 1.0)
    aspect_ratio = 1.0: circle
    aspect_ratio > 1.0: ellipse (elongated along x)
    
    Total area is kept approximately constant by adjusting radii:
    Area = π * rx * ry = constant
    """
    edges = np.zeros((ny, nx), dtype=np.bool_)
    cx, cy = (nx - 1) / 2, (ny - 1) / 2
    
    # Keep area constant: rx * ry = constant
    # For aspect_ratio = 1: rx = ry = min(cx, cy)
    # For aspect_ratio > 1: rx increases, ry decreases proportionally
    base_radius = min(cx, cy)
    rx = base_radius * np.sqrt(aspect_ratio)
    ry = base_radius / np.sqrt(aspect_ratio)
    
    for y in range(ny):
        for x in range(nx):
            dx = (x - cx) / rx if rx > 0 else 0
            dy = (y - cy) / ry if ry > 0 else 0
            if dx * dx + dy * dy >= 1.0:
                edges[y, x] = True
    return edges

def calculate_grid_size_for_constant_area(base_nx, base_ny, aspect_ratio, target_area):
    """
    Use fixed grid size to avoid discretization issues.
    The ellipse itself maintains constant area through rx * ry = constant.
    By keeping the grid fixed, we ensure fair comparison.
    """
    # Keep grid size constant - the ellipse area is maintained by the radius formula
    return base_nx, base_ny

# ------------------------------
# Relaxation Solver
# ------------------------------
@njit(cache=True, fastmath=True)
def _relax(height, edges, dot_y, dot_x, dot_height, max_iter=50000, tol=1e-8):
    ny, nx = height.shape
    for _ in range(max_iter):
        max_diff = 0.0
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                if x == dot_x and y == dot_y:
                    height[y, x] = dot_height
                    continue
                if edges[y, x]:
                    height[y, x] = 0.0
                    continue
                new_val = 0.25 * (
                    height[y + 1, x] +
                    height[y - 1, x] +
                    height[y, x + 1] +
                    height[y, x - 1]
                )
                diff = abs(new_val - height[y, x])
                if diff > max_diff:
                    max_diff = diff
                height[y, x] = new_val
        if max_diff < tol:
            break
    return height

def surface(nx, ny, dot_pos, dot_height, edges):
    height = np.zeros((ny, nx), dtype=np.float64)
    dot_y, dot_x = dot_pos
    return _relax(height, edges, dot_y, dot_x, dot_height)
