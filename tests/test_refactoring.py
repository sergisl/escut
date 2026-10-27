import pytest


def test_submodule_imports():
    # Importing modules should not raise
    import boundary_conditions
    import mesh_utils
    import physics_utils
    import core_solver


def test_unified_import_point_functions_exist():
    import escut as Qsolver  # noqa: N813

    functions = [
        'solve_scalar_bvp',
        'scalar_equation_and_jac',
        'smoothed_density',
        'Q_small_r',
        'make_initial_guess_from_analytic',
        'build_clustered_mesh',
        'outer_bc_robin_coeffs',
        'bc_robin',
        'bc_robin_jac',
        # charge formulation (added in this release)
        'solve_scalar_bvp_charge',
        'solve_scalar_bvp_charge_log',
        'charge_discriminant',
        'charge_gradient',
    ]

    for fname in functions:
        assert hasattr(Qsolver, fname), f"Qsolver missing {fname}"


def test_functional_behavior_quick():
    import numpy as np
    import physics_utils
    import mesh_utils
    import boundary_conditions
    import escut as Qsolver  # noqa: N813

    # physics_utils
    Q_in = physics_utils.Q_small_r(B=-0.1, C=-1, S=1, delta_c=1.0)
    assert isinstance(Q_in, float)

    # mesh_utils
    x_mesh = mesh_utils.build_clustered_mesh(1e-3, 50, R=1.0, n_inner=50, n_shell=100, n_outer=50)
    assert hasattr(x_mesh, '__len__') and len(x_mesh) > 100

    # boundary_conditions
    k = boundary_conditions.outer_bc_robin_coeffs(A=1, B=-0.1, C=-1, xmax=50, Q_inf=0)
    assert isinstance(k, float)

    # core solver via backward compat wrapper (quick run)
    sol = Qsolver.solve_scalar_bvp(
        A=1, B=-0.1, C=-1, S=1, delta_c=1.0,
        xmin=1e-3, xmax=50, npoints=200,
        homotopy=False,
    )
    assert hasattr(sol, 'success') and sol.success is True


def test_modular_benefits_and_docs():
    from physics_utils import Q_small_r, make_initial_guess_from_analytic
    from mesh_utils import build_clustered_mesh
    import escut as Qsolver  # noqa: N813

    assert callable(Q_small_r)
    assert callable(make_initial_guess_from_analytic)
    assert callable(build_clustered_mesh)

    # Check that Qsolver's exported docstring for Q_small_r is non-trivial
    assert Qsolver.Q_small_r.__doc__ and len(Qsolver.Q_small_r.__doc__) > 50
