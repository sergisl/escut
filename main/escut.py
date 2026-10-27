"""Main import surface for the scalar BVP solver: use `from escut import ...` in notebooks and scripts."""

# Core solvers
from core_solver import (
    scalar_equation_and_jac,
    solve_scalar_bvp,
)

# First-order (Q, I) formulation
# (method="charge" in solve_scalar_bvp above dispatches here)
from charge_solver import (
    solve_scalar_bvp_charge,
    solve_scalar_bvp_charge_log,
    charge_gradient,
    charge_discriminant,
    charge_from_flux,
    charge_vainshtein_radius,
    background_mass,
)

# Physics utilities
from physics_utils import (
    smoothed_density,
    Q_small_r,
    V_eff,
    make_initial_guess_from_analytic,
    mu_for_tanh_step,
    _cross_at_one,
    compute_n_slope,
    analytic_Q_thin_shell,
    enclosed_source,
    vainshtein_flux,
    vainshtein_radius,
    make_vainshtein_solution,
    effective_gravity,
    screening_factor,
)

# Mesh utilities
from mesh_utils import (
    build_clustered_mesh,
)

# Boundary conditions
from boundary_conditions import (
    outer_bc_robin_coeffs,
    bc_robin,
    bc_robin_jac,
)

# Plotting utilities (support both package and module contexts)
try:
    from .plotting import (
        plot_double_panel,
        plot_profiles_and_effective_mass_combined,
        plot_effective_potential,
    )
except ImportError:
    from plotting import (
        plot_double_panel,
        plot_profiles_and_effective_mass_combined,
        plot_effective_potential,
    )

# For convenience, expose key submodule names
import core_solver as core_solver
import charge_solver as charge_solver
import physics_utils as physics_utils
import mesh_utils as mesh_utils
import boundary_conditions as boundary_conditions
import plotting as plotting
import styling as styling

__all__ = [
    # Core solvers
    'scalar_equation_and_jac',
    'solve_scalar_bvp',

    # Charge-form (Q, 𝓘) solver
    'solve_scalar_bvp_charge',
    'solve_scalar_bvp_charge_log',
    'charge_gradient',
    'charge_discriminant',
    'charge_from_flux',
    'charge_vainshtein_radius',
    'background_mass',

    # Physics utilities
    'smoothed_density',
    'Q_small_r',
    'make_initial_guess_from_analytic',
    'mu_for_tanh_step',
    'analytic_Q_thin_shell',
    'enclosed_source',
    'vainshtein_flux',
    'vainshtein_radius',
    'make_vainshtein_solution',
    'effective_gravity',
    'screening_factor',

    # Mesh utilities
    'build_clustered_mesh',

    # Boundary conditions
    'outer_bc_robin_coeffs',
    'bc_robin',
    'bc_robin_jac',

    # Submodules
    'core_solver',
    'charge_solver',
    'physics_utils',
    'mesh_utils',
    'boundary_conditions',
    'plotting',
    'styling',

    # Plotting exports
    'compute_n_slope',
    'plot_double_panel',
    'plot_profiles_and_effective_mass_combined',
    # Geometry utilities
    '_cross_at_one',
]
