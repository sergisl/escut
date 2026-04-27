"""Mesh construction: build_clustered_mesh generates a 1D grid refined around the object boundary."""

import numpy as np
from typing import Optional


def build_clustered_mesh(xmin: float, xmax: float, R: float,
                         n_inner: int = 200, n_shell: int = 800, n_outer: int = 200,
                         shell_width: float = 0.05, power_inner: float = 1.0,
                         power_outer: float = 1.0) -> np.ndarray:
    """Build a 1D mesh denser in a shell around R = object boundary.

    Three regions: coarse inner [xmin, R-shell_width], dense shell
    [R±shell_width], coarse outer [R+shell_width, xmax].

    Parameters
    ----------
    xmin, xmax : float   Domain bounds.
    R : float            Object radius (typically 1.0).
    n_inner, n_shell, n_outer : int  Points per region.
    shell_width : float  Half-width of dense shell (default 0.05).
    power_inner, power_outer : float  Power-law spacing exponents (1.0 = uniform).

    Returns
    -------
    mesh : ndarray  Sorted unique grid, strictly increasing.
    """
    # Ensure reasonable bounds
    left = max(xmin, R - 5.0*shell_width)
    right = min(xmax, R + 5.0*shell_width)
    
    # Inner region (coarser, with power law spacing)
    r1 = np.linspace(xmin, max(xmin, R - shell_width), n_inner)**power_inner
    
    # Dense shell (uniform spacing)
    r2 = np.linspace(max(xmin, R - shell_width), min(xmax, R + shell_width), n_shell)
    
    # Outer region (coarser, with power law spacing)
    r3 = np.linspace(min(xmax, R + shell_width), xmax, n_outer)**power_outer
    
    # Combine and clean up
    mesh = np.unique(np.concatenate([r1, r2, r3]))
    
    # Ensure strictly increasing and within bounds
    mesh = mesh[(mesh >= xmin) & (mesh <= xmax)]
    mesh.sort()
    
    return mesh
