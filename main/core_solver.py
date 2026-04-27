"""BVP solver for the unified screening equation: scalar_equation_and_jac and solve_scalar_bvp."""

import numpy as np
from scipy.integrate import solve_bvp
from typing import Callable, Tuple, Optional, Dict, Any

from physics_utils import smoothed_density
from boundary_conditions import outer_bc_robin_coeffs, bc_robin, bc_robin_jac


def scalar_equation_and_jac(x: np.ndarray, y: np.ndarray, pars: np.ndarray, 
                            delta_x: Callable, eps_den: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """RHS and Jacobian for the unified screening ODE, Qxx = N/D.

    N = -S x² δ - (A+DQ)(2x Qx) - x²Q(B+CQ) + 2F Qx² - E x² Qx²
    D = (A+DQ) x² - 4F x Qx

    Parameters
    ----------
    x : ndarray          Spatial grid.
    y : ndarray (2, N)   State [Q, Qx].
    pars : ndarray (7,)  Coefficients [A, B, C, D, E, F, S].
    delta_x : callable   Density profile δ(x).
    eps_den : float      Denominator regularisation floor (default 1e-12).

    Returns
    -------
    f : ndarray (2, N)      RHS [Qx, Qxx].
    jac : ndarray (2, 2, N) ∂f/∂(Q, Qx).
    """
    Q, Qx = y
    A, B, C, D, E, F, S = pars

    # Numerator N (all terms except those multiplying Qxx)
    N  = (-S * delta_x(x)) * x**2 \
         - (A + D*Q) * (2*x*Qx) \
         - x**2 * Q * (B + C*Q) \
         + 2*F * Qx**2 \
         - E * x**2 * Qx**2

    # Denominator D (coefficient of Qxx)
    Dn = (A + D*Q) * x**2 - 4*F * x * Qx

    # Scale-aware regularization to avoid division blow-ups when Dn ~ 0
    # We compute a characteristic scale from all contributing terms
    scale   = np.abs(A)*x**2 + np.abs(D)*np.abs(Q)*x**2 + 4*np.abs(F)*x*np.abs(Qx) + 1.0
    Dn_safe = np.where(np.abs(Dn) < eps_den*scale, np.sign(Dn)*eps_den*scale, Dn)

    Qxx = N / Dn_safe
    f   = np.vstack((Qx, Qxx))

    # ---- Analytic Jacobian: ∂f/∂(Q, Qx)
    dN_dQ  = -(2*x*Qx)*D - x**2*(B + 2*C*Q)
    dN_dQx = -(A + D*Q)*(2*x) + 4*F*Qx - 2*E*x**2*Qx
    dD_dQ  = D * x**2
    dD_dQx = -4 * F * x

    invD2    = 1.0 / (Dn_safe**2)
    dQxx_dQ  = (dN_dQ  * Dn_safe - N * dD_dQ)  * invD2
    dQxx_dQx = (dN_dQx * Dn_safe - N * dD_dQx) * invD2

    jac = np.zeros((2, 2, x.size))
    jac[0, 1, :] = 1.0
    jac[1, 0, :] = dQxx_dQ
    jac[1, 1, :] = dQxx_dQx
    return f, jac


def solve_scalar_bvp(A: float = 1, B: float = 0, C: float = 0, D: float = 0, 
                     E: float = 0, F: float = 0, *,
                     S: float = 1.0,
                     delta_c: float = 1.0, delta_inf: float = 0.0, eps_edge: float = 1e-3,
                     Q_inf: float = 0.0,
                     xmin: float = 1e-3, xmax: float = 50.0, npoints: int = 800,
                     homotopy: bool = True, steps: int = 10,
                     tol: float = 1e-7, max_nodes: int = 800000,
                     asymptotic_bc: bool = True,
                     warn_exterior_mu_ratio: float = 1e-3) -> Any:
    """Solve the spherical scalar BVP for Q(x), x = r/R.

    Uses homotopy continuation by default: ramps nonlinear coefficients (C,D,E,F)
    from 0 to target over `steps` steps, using each solution as the next initial guess.

    Parameters
    ----------
    A, B, C, D, E, F : float  Equation coefficients.
    S : float                  Source amplitude (default 1.0).
    delta_c, delta_inf : float Overdensities inside/outside object.
    eps_edge : float           Density transition width (default 1e-3).
    Q_inf : float              Asymptotic Q value (default 0.0).
    xmin, xmax : float         Domain bounds.
    npoints : int              Initial mesh size (default 800).
    homotopy : bool            Enable homotopy continuation (default True).
    steps : int                Number of homotopy steps (default 10).
    tol : float                BVP solver tolerance (default 1e-7).
    max_nodes : int            Maximum refinement nodes (default 800000).

    Returns
    -------
    sol : BVPResult  scipy solution with added `.meta` dict of problem parameters.
    """
    delta_x = lambda xx: smoothed_density(xx, delta_c, delta_inf, eps=eps_edge)
    x = np.linspace(xmin, xmax, npoints)
    y_guess = np.vstack((Q_inf*np.ones_like(x), np.zeros_like(x)))

    # Optionally compute an analytic asymptotic match for the outer BC
    Q_inf_use = float(Q_inf)
    if asymptotic_bc:
        try:
            from physics_utils import _mu_eff_Q_in_m_eff
            mu_eff, Q_in_est, m_eff = _mu_eff_Q_in_m_eff(A, B, C, S, delta_c, R=1.0)
            # estimate Q at xmax from Yukawa-like matched exterior (see mu_eff_Q_in_m_eff)
            # guard against non-finite outputs from mu_eff_Q_in_m_eff
            if not np.isfinite(mu_eff) or not np.isfinite(m_eff):
                Q_inf_use = float(Q_inf)
            else:
                if m_eff > 0:
                    Q_inf_use = (mu_eff / (8.0 * np.pi * A * xmax)) * np.exp(-m_eff * (xmax - 1.0)) / (1.0 + m_eff * 1.0)
                else:
                    Q_inf_use = (mu_eff / (8.0 * np.pi * A * xmax)) if A != 0 else float(Q_inf)
        except Exception:
            Q_inf_use = float(Q_inf)

    k = outer_bc_robin_coeffs(A, B, C, xmax, Q_inf_use)

    def make_problem(pars):
        def fun(xx, yy):     return scalar_equation_and_jac(xx, yy, pars, delta_x)[0]
        def fun_jac(xx, yy): return scalar_equation_and_jac(xx, yy, pars, delta_x)[1]
        def bc(ya, yb):      return bc_robin(ya, yb, Q_inf_use, k)
        def bc_jac(ya, yb):  return bc_robin_jac(ya, yb, Q_inf_use, k)
        return fun, fun_jac, bc, bc_jac

    pars_lin  = np.array([A, B, 0, 0, 0, 0, S], dtype=float)
    pars_full = np.array([A, B, C, D, E, F, S], dtype=float)

    fun, fun_jac, bc, bc_jac = make_problem(pars_lin)
    sol = solve_bvp(fun, bc, x, y_guess, fun_jac=fun_jac, bc_jac=bc_jac,
                    tol=tol, max_nodes=max_nodes)

    if homotopy:
        for i in range(1, steps + 1):
            lam = i / steps
            pars_h = pars_lin.copy()
            pars_h[2:] = pars_lin[2:] + lam * (pars_full[2:] - pars_lin[2:])
            fun, fun_jac, bc, bc_jac = make_problem(pars_h)
            sol = solve_bvp(fun, bc, sol.x, sol.y, fun_jac=fun_jac, bc_jac=bc_jac,
                            tol=tol, max_nodes=max_nodes)
            if not sol.success:
                print(f"[warn] Homotopy step {i} failed: {sol.message}")
                break
    else:
        fun, fun_jac, bc, bc_jac = make_problem(pars_full)
        sol = solve_bvp(fun, bc, x, y_guess, fun_jac=fun_jac, bc_jac=bc_jac,
                        tol=tol, max_nodes=max_nodes)

    sol.meta = dict(
        A=A, B=B, C=C, D=D, E=E, F=F, S=S,
        xmin=xmin, xmax=xmax,
        delta_c=delta_c, delta_inf=delta_inf, eps_edge=eps_edge,
        Q_inf=Q_inf_use, k=k,
        pars_full=pars_full, pars_lin=pars_lin,
        indep_var='x'
    )
    return sol