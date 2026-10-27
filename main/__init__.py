"""escut: scalar BVP solver utilities for screening mechanisms.

Mirrors the public surface of `escut.py`, so both `import escut` (installed
package) and a flat `sys.path` import of `main/` expose the same names.
"""

from .core_solver import (
    scalar_equation_and_jac,
    solve_scalar_bvp,
)

from .charge_solver import (
    solve_scalar_bvp_charge,
    solve_scalar_bvp_charge_log,
    charge_gradient,
    charge_discriminant,
    charge_from_flux,
    charge_vainshtein_radius,
    background_mass,
)

from .physics_utils import (
    smoothed_density,
    Q_small_r,
    V_eff,
    make_initial_guess_from_analytic,
    mu_for_tanh_step,
    compute_n_slope,
    analytic_Q_thin_shell,
    enclosed_source,
    vainshtein_flux,
    vainshtein_radius,
    make_vainshtein_solution,
    effective_gravity,
    screening_factor,
)

from .mesh_utils import build_clustered_mesh

from .boundary_conditions import (
    outer_bc_robin_coeffs,
    bc_robin,
    bc_robin_jac,
)

from . import styling

from .plotting import (
    plot_double_panel,
    plot_profiles_and_effective_mass_combined,
    plot_effective_potential,
)

__all__ = [
    'scalar_equation_and_jac', 'solve_scalar_bvp',
    'solve_scalar_bvp_charge', 'solve_scalar_bvp_charge_log',
    'charge_gradient', 'charge_discriminant', 'charge_from_flux',
    'charge_vainshtein_radius', 'background_mass',
    'smoothed_density', 'Q_small_r', 'V_eff',
    'make_initial_guess_from_analytic', 'mu_for_tanh_step', 'compute_n_slope',
    'analytic_Q_thin_shell', 'enclosed_source', 'vainshtein_flux',
    'vainshtein_radius', 'make_vainshtein_solution', 'effective_gravity',
    'screening_factor',
    'build_clustered_mesh',
    'outer_bc_robin_coeffs', 'bc_robin', 'bc_robin_jac',
    'plot_double_panel', 'plot_profiles_and_effective_mass_combined',
    'plot_effective_potential',
    'styling',
]
