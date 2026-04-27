"""
Escut package

Lightweight scalar BVP solver utilities for screening mechanisms.

This package exposes the same public API surfaced via the `escut.py`
module, allowing `import escut` in addition to `from escut import ...`.
"""

from .core_solver import (
    scalar_equation_and_jac,
    solve_scalar_bvp,
)

from .physics_utils import (
    smoothed_density,
    Q_small_r,
    mu_eff_Q_in_m_eff,
    make_initial_guess_from_analytic,
    mu_for_tanh_step,
    mu_cumulative_for_tanh_step,
)

from .mesh_utils import (
    build_clustered_mesh,
)

from .boundary_conditions import (
    outer_bc_robin_coeffs,
    bc_robin,
    bc_robin_jac,
)

__all__ = [
    # Core solvers
    'scalar_equation_and_jac',
    'solve_scalar_bvp',

    # Physics utilities
    'smoothed_density',
    'Q_small_r',
    'mu_eff_Q_in_m_eff',
    'make_initial_guess_from_analytic',
    'mu_for_tanh_step',
    'mu_cumulative_for_tanh_step',

    # Mesh utilities
    'build_clustered_mesh',

    # Boundary conditions
    'outer_bc_robin_coeffs',
    'bc_robin',
    'bc_robin_jac',
]
