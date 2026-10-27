"""First-order (Q, I) formulation of the unified screening equation: an
alternative to the second-order (Q, Qx) form in core_solver.

Integrating the kinetic sector once gives the enclosed charge I(x),

    I  = x^2 (A + D Q) Qx - 2F x Qx^2
    I' = -x^2 [ S delta(x) + Q(B + C Q + G Q^2) ]

The I equation has no denominator that can vanish, and Qx is recovered from I
algebraically (a quadratic in Qx when F != 0). Exact when E == D.
"""

import numpy as np
from scipy.integrate import solve_bvp, cumulative_trapezoid
from typing import Callable, Optional

try:  # package import; falls back to flat when main/ is on sys.path
    from .physics_utils import (smoothed_density, make_initial_guess_from_analytic,
                                vainshtein_flux, _mu_eff_Q_in_m_eff)
    from .boundary_conditions import outer_bc_robin_coeffs
except ImportError:
    from physics_utils import (smoothed_density, make_initial_guess_from_analytic,
                               vainshtein_flux, _mu_eff_Q_in_m_eff)
    from boundary_conditions import outer_bc_robin_coeffs


# ---------------------------------------------------------------------------
# Charge <-> flux algebra
# ---------------------------------------------------------------------------

def charge_gradient(x: np.ndarray, Q: np.ndarray, Icharge: np.ndarray,
                    A: float, D: float, F: float, eps_den: float = 1e-12) -> np.ndarray:
    """Flux Qx(x) recovered algebraically from the charge-form state (Q, I).

    Takes the branch continuous with the F -> 0 limit Qx -> I/((A + D Q) x^2).

    Parameters
    ----------
    x, Q, Icharge : ndarray  Coordinate and state (broadcastable shapes).
    A, D, F : float          Kinetic, Phaedrus-braiding and Vainshtein coefficients.
    eps_den : float          Regularisation floor for the F = 0 branch.

    Returns
    -------
    Qx : ndarray
    """
    x = np.asarray(x, dtype=float)
    Q = np.asarray(Q, dtype=float)
    Icharge = np.asarray(Icharge, dtype=float)
    Aeff = A + D * Q
    if F == 0:
        scale = np.abs(Aeff) * x**2 + 1.0
        denom = np.where(np.abs(Aeff) * x**2 < eps_den * scale,
                         np.sign(Aeff) * eps_den * scale + (Aeff == 0) * eps_den * scale,
                         Aeff * x**2)
        return Icharge / denom
    disc = charge_discriminant(x, Q, Icharge, A, D, F)
    safe_x = np.where(np.abs(x) < 1e-300, 1e-300, x)
    return (Aeff * x**2 - np.sqrt(np.maximum(disc, 0.0))) / (4.0 * F * safe_x)


def charge_discriminant(x: np.ndarray, Q: np.ndarray, Icharge: np.ndarray,
                        A: float, D: float, F: float) -> np.ndarray:
    """Discriminant Δ = (A + D Q)² x⁴ − 8F x I of the quadratic for Qx.

    Δ < 0 means no real Qx exists at that state: the imaginary fifth-force
    pathology that arises in sufficiently underdense regions.
    """
    x = np.asarray(x, dtype=float)
    Q = np.asarray(Q, dtype=float)
    Icharge = np.asarray(Icharge, dtype=float)
    return (A + D * Q)**2 * x**4 - 8.0 * F * x * Icharge


def charge_vainshtein_radius(x: np.ndarray, Icharge: np.ndarray,
                             A: float, F: float) -> float:
    """Vainshtein radius: outermost root of A² x³ = 8|F I(x)|, where the
    nonlinear term overtakes the linear one in magnitude.

    Compares magnitudes, so it is independent of the sign conventions chosen for
    F and I. Distinct from `charge_discriminant`'s zero-crossing, which instead
    flags the loss of a real Qx. Returns NaN if F = 0 or no crossing is
    bracketed by `x`.
    """
    x = np.asarray(x, dtype=float)
    Icharge = np.asarray(Icharge, dtype=float)
    if F == 0:
        return float("nan")
    g = A**2 * x**3 - 8.0 * abs(F) * np.abs(Icharge)
    s = np.sign(g)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if idx.size == 0:
        return float("nan")
    j = idx[-1]
    x0, x1, g0, g1 = x[j], x[j + 1], g[j], g[j + 1]
    return float(x0 + (x1 - x0) * (-g0) / (g1 - g0))


def charge_from_flux(x: np.ndarray, Q: np.ndarray, Qx: np.ndarray,
                     A: float, D: float, F: float) -> np.ndarray:
    """Inverse map: the charge I(x) implied by a known (Q, Qx) profile.

    Used to build an initial guess consistent with a given Q(x); seeding I = 0
    against a varying Q contradicts the state and stalls the solver.
    """
    x = np.asarray(x, dtype=float)
    Q = np.asarray(Q, dtype=float)
    Qx = np.asarray(Qx, dtype=float)
    return x**2 * (A + D * Q) * Qx - 2.0 * F * x * Qx**2


def background_mass(source_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                    Q_bg: float, x_outer: float, A: float) -> float:
    """Yukawa decay rate m of fluctuations about Q_bg, for the outer Robin BC.

    Linearising A ∇²Q = -source_fn(x, Q) about the background gives
    m² = -source_fn'(Q_bg) / A, with the derivative taken numerically. Returns
    0 for a massless background (m² <= 0), i.e. a pure 1/x tail.
    """
    h = 1e-5 * max(abs(Q_bg), 1.0)
    xb = np.full(1, float(x_outer))
    dS = (np.asarray(source_fn(xb, np.full(1, Q_bg + h)), dtype=float)
          - np.asarray(source_fn(xb, np.full(1, Q_bg - h)), dtype=float)) / (2.0 * h)
    m2 = -float(dS[0]) / A
    return float(np.sqrt(m2)) if m2 > 0 else 0.0


# ---------------------------------------------------------------------------
# Solution containers
# ---------------------------------------------------------------------------

class ChargeBVPResult:
    """Solved (Q, I) profile presented through the scipy BVPResult surface.

    The solver's state is (Q, I), but `.sol(x)` returns [Q, Qx] so consumers
    see the same layout as `solve_scalar_bvp`.
    """

    def __init__(self, sol, A, D, F, success, message, meta):
        self._interp = sol.sol                      # scipy interpolant, x -> [Q, I]
        self.x = sol.x                              # final adaptive mesh
        self.y = np.vstack([sol.y[0],               # [Q, Qx] on that mesh
                            charge_gradient(sol.x, sol.y[0], sol.y[1], A, D, F)])
        self.success = success
        self.message = message
        self.meta = meta
        self._A, self._D, self._F = A, D, F         # to redo I -> Qx in .sol()

    def sol(self, x):
        x = np.asarray(x, dtype=float)
        Q, Icharge = self._interp(x)
        return np.vstack([Q, charge_gradient(x, Q, Icharge, self._A, self._D, self._F)])

    def discriminant(self, x):
        """Δ(x) on the solved profile; see `charge_discriminant`."""
        x = np.asarray(x, dtype=float)
        Q, Icharge = self._interp(x)
        return charge_discriminant(x, Q, Icharge, self._A, self._D, self._F)


class LogChargeBVPResult(ChargeBVPResult):
    """As `ChargeBVPResult`, for a solve carried out in u = ln Q."""

    def __init__(self, sol, A, D, F, meta):
        Q_mesh = np.exp(np.clip(sol.y[0], -60.0, 60.0))
        self._interp = sol.sol                      # x -> [u, I]
        self.x = sol.x
        self.y = np.vstack([Q_mesh,
                            charge_gradient(sol.x, Q_mesh, sol.y[1], A, D, F)])
        self.success = sol.success
        self.message = sol.message
        self.meta = meta
        self._A, self._D, self._F = A, D, F

    def _state(self, x):
        u, Icharge = self._interp(np.asarray(x, dtype=float))
        return np.exp(np.clip(u, -60.0, 60.0)), Icharge

    def sol(self, x):
        x = np.asarray(x, dtype=float)
        Q, Icharge = self._state(x)
        return np.vstack([Q, charge_gradient(x, Q, Icharge, self._A, self._D, self._F)])

    def discriminant(self, x):
        x = np.asarray(x, dtype=float)
        Q, Icharge = self._state(x)
        return charge_discriminant(x, Q, Icharge, self._A, self._D, self._F)


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def solve_scalar_bvp_charge(A: float = 1, B: float = 0, C: float = 0, D: float = 0,
                            E: float = 0, F: float = 0, G: float = 0, *,
                            S: float = 1.0,
                            delta_c: float = 1.0, delta_inf: float = 0.0,
                            eps_edge: float = 1e-3, Q_inf: float = 0.0,
                            source_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
                            guess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                            xmin: float = 1e-3, xmax: float = 50.0, npoints: int = 800,
                            homotopy: bool = True, steps: int = 10,
                            tol: float = 1e-7, max_nodes: int = 800000,
                            asymptotic_bc: bool = True,
                            x_init: Optional[np.ndarray] = None,
                            m_bg: Optional[float] = None,
                            seed: Optional[tuple] = None) -> ChargeBVPResult:
    """Solve the spherical scalar BVP for Q(x) in the charge form (Q, I).

    Same coefficients, density profile and return surface as
    `core_solver.solve_scalar_bvp`, but integrates the first integral of the
    kinetic sector so no denominator can vanish. Requires E == D.

    Parameters
    ----------
    A, B, C, D, E, F, G : float  Equation coefficients (as in `solve_scalar_bvp`).
    S, delta_c, delta_inf, eps_edge, Q_inf : as in `solve_scalar_bvp`.
    source_fn : callable, optional
        source_fn(x, Q) replacing the polynomial potential Q(B + C Q + G Q²),
        so B, C, G are ignored. Q is then the full field rather than a
        perturbation about Q_inf. The S·delta(x) term is still added, so pass
        S = 0 when source_fn already carries the density dependence.
    guess_fn : callable, optional
        guess_fn(x) -> Q(x) initial-guess profile. Defaults to the analytic
        thin-shell guess, or a flat Q_inf when `source_fn` is given.
    xmin, xmax, npoints, homotopy, steps, tol, max_nodes : as in `solve_scalar_bvp`.
    asymptotic_bc : bool
        Replace the outer pin Q_inf by an analytic Yukawa-tail estimate at
        xmax, since Q(xmax) != Q_inf at finite xmax. Polynomial mode only.
    x_init : ndarray, optional
        Initial mesh, overriding the default linear grid. A log grid resolves a
        screened interior far better; see `mesh_utils.build_clustered_mesh`.
    m_bg : float, optional
        Yukawa decay rate for the outer Robin BC. Defaults to the polynomial
        estimate, or to `background_mass(source_fn, ...)` in full-field mode.
    seed : (x, Q, I), optional
        Previous solution to warm-start from, interpolated onto the mesh. The
        homotopy ramp is then skipped, since re-ramping discards the seed.

    Returns
    -------
    sol : ChargeBVPResult
    """
    if not np.isclose(E, D):
        raise ValueError(
            f"solve_scalar_bvp_charge requires E == D (got D={D}, E={E}); the "
            "first integral is exact only when they match. Use "
            "solve_scalar_bvp(..., method='gradient') otherwise."
        )

    delta_x = lambda xx: smoothed_density(xx, delta_c, delta_inf, eps=eps_edge)
    x = (np.asarray(x_init, dtype=float) if x_init is not None
         else np.linspace(xmin, xmax, npoints))
    x_out = float(x[-1])

    # ---- initial guess: a Q(x) profile and the I(x) it implies ----
    if seed is not None:
        xs, Qs, Is = seed
        Q_guess = np.interp(x, xs, Qs)
        I_guess = np.interp(x, xs, Is)
    else:
        if guess_fn is not None:
            Q_guess = guess_fn(x)
        elif source_fn is None and (B != 0 or C != 0 or G != 0):
            Q_guess = make_initial_guess_from_analytic(x, A, B, C, S, delta_c, R=1.0, G=G)[0]
        elif source_fn is None:
            # B = C = G = 0: no interior minimum for the thin-shell guess to use,
            # so seed from the exact algebraic Vainshtein flux instead.
            Qx_seed = vainshtein_flux(x, A, F, delta_c, delta_inf, eps_edge, S)
            cumQ = cumulative_trapezoid(Qx_seed, x, initial=0.0)
            Q_guess = cumQ - cumQ[-1] + Q_inf
        else:
            Q_guess = np.full_like(x, Q_inf)
        Q_guess = np.asarray(Q_guess, dtype=float)
        I_guess = charge_from_flux(x, Q_guess, np.gradient(Q_guess, x), A, D, F)
    Y = np.vstack([Q_guess, I_guess])

    # ---- outer Robin BC ----
    Q_inf_use = float(Q_inf)
    if source_fn is not None:
        # Full-field mode: the polynomial route cannot see the potential, so take
        # the decay rate from source_fn itself.
        m_out = background_mass(source_fn, Q_inf, x_out, A) if m_bg is None else float(m_bg)
        k = m_out + 1.0 / x_out
    else:
        if asymptotic_bc and A != 0:
            try:
                mu_eff, _, m_eff = _mu_eff_Q_in_m_eff(A, B, C, S, delta_c, R=1.0, G=G)
                if np.isfinite(mu_eff) and np.isfinite(m_eff):
                    Q_inf_use = mu_eff / (8.0 * np.pi * A * x_out)
                    if m_eff > 0:
                        Q_inf_use *= np.exp(-m_eff * (x_out - 1.0)) / (1.0 + m_eff)
            except Exception:
                Q_inf_use = float(Q_inf)
        k = (outer_bc_robin_coeffs(A, B, C, x_out, Q_inf_use, G=G) if m_bg is None
             else float(m_bg) + 1.0 / x_out)

    def potential(xx, Q, lam):
        # B stays at full strength and only C, G ramp, matching the homotopy
        # split in solve_scalar_bvp.
        if source_fn is not None:
            return source_fn(xx, Q)
        return Q * B + lam * (C * Q**2 + G * Q**3)

    def make_problem(Fk, Dk, lam):
        def fun(xx, YY):
            Q, Icharge = YY
            Qx = charge_gradient(xx, Q, Icharge, A, Dk, Fk)
            return np.vstack([Qx, -xx**2 * (S * delta_x(xx) + potential(xx, Q, lam))])

        def bc(Ya, Yb):
            Qb, Ib = Yb
            Qxb = charge_gradient(np.array([x_out]), np.array([Qb]), np.array([Ib]),
                                  A, Dk, Fk)[0]
            return np.array([Ya[1], Qxb + k * (Qb - Q_inf_use)])
        return fun, bc

    if homotopy and seed is None:
        x_cur, Y_cur = x, Y
        for i in range(1, steps + 1):
            lam = i / steps
            fun, bc = make_problem(lam * F, lam * D, lam)
            sol = solve_bvp(fun, bc, x_cur, Y_cur, tol=tol, max_nodes=max_nodes, verbose=0)
            x_cur, Y_cur = sol.x, sol.y
            if not sol.success:
                print(f"[warn] charge-form homotopy step {i} failed: {sol.message}")
                break
    else:
        fun, bc = make_problem(F, D, 1.0)
        sol = solve_bvp(fun, bc, x, Y, tol=tol, max_nodes=max_nodes, verbose=0)

    meta = dict(A=A, B=B, C=C, D=D, E=E, F=F, G=G, S=S,
                xmin=float(x[0]), xmax=x_out,
                delta_c=delta_c, delta_inf=delta_inf, eps_edge=eps_edge,
                Q_inf=Q_inf_use, k=k, indep_var='x', formulation='charge')
    return ChargeBVPResult(sol, A, D, F, sol.success, sol.message, meta)


def solve_scalar_bvp_charge_log(A: float = 1, D: float = 0, E: float = 0, F: float = 0, *,
                                S: float = 1.0, delta_c: float = 1.0, delta_inf: float = 0.0,
                                eps_edge: float = 1e-3, Q_bg: float,
                                source_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                                guess_fn: Callable[[np.ndarray], np.ndarray],
                                xmin: float = 1e-2, xmax: float = 1e3, npoints: int = 1500,
                                homotopy: bool = True, steps: int = 6,
                                tol: float = 1e-7, max_nodes: int = 200000,
                                x_init: Optional[np.ndarray] = None,
                                m_bg: Optional[float] = None,
                                outer_bc: str = "robin",
                                seed: Optional[tuple] = None) -> LogChargeBVPResult:
    """Solve in u = ln Q instead of Q, for deeply screened profiles.

    A screened core or void spans many decades in Q, which a linear mesh cannot
    resolve; in u it is an O(1) descent. Requires Q > 0 throughout, so Q must be
    a field amplitude rather than a perturbation that can change sign.

    Parameters
    ----------
    A, D, E, F : float  Kinetic, Phaedrus and Vainshtein coefficients (E == D).
    S, delta_c, delta_inf, eps_edge : as elsewhere; pass S = 0 when `source_fn`
        already carries the density dependence.
    Q_bg : float        Asymptotic value of Q at xmax, Q_bg > 0.
    source_fn : callable  source_fn(x, Q); required here (Q is the full field).
    guess_fn : callable   guess_fn(x) -> Q(x) initial guess, strictly positive.
    xmin, xmax, npoints, homotopy, steps, tol, max_nodes : as elsewhere.
    x_init : ndarray, optional
        Initial mesh, overriding the default log-spaced grid. A thin screening
        shell needs nodes clustered near x = 1 to converge within `max_nodes`;
        see `mesh_utils.build_clustered_mesh`.
    m_bg : float, optional
        Yukawa decay rate for the Robin BC; defaults to
        `background_mass(source_fn, Q_bg, xmax, A)`.
    outer_bc : {"robin", "dirichlet"}
        "robin" lets the field relax at xmax; "dirichlet" pins Q(xmax) = Q_bg,
        which distorts the slope there unless xmax is far outside the range.
    seed : (x, Q, I), optional
        Previous solution to warm-start from; skips the homotopy ramp.

    Returns
    -------
    sol : LogChargeBVPResult
    """
    if not np.isclose(E, D):
        raise ValueError(
            f"solve_scalar_bvp_charge_log requires E == D (got D={D}, E={E}); "
            "see solve_scalar_bvp_charge."
        )
    if Q_bg <= 0:
        raise ValueError(f"solve_scalar_bvp_charge_log requires Q_bg > 0 (got {Q_bg}).")

    delta_x = lambda xx: smoothed_density(xx, delta_c, delta_inf, eps=eps_edge)
    x = (np.asarray(x_init, dtype=float) if x_init is not None
         else np.logspace(np.log10(xmin), np.log10(xmax), npoints))
    x_out = float(x[-1])

    if seed is not None:
        xs, Qs, Is = seed
        Y = np.vstack([np.interp(x, xs, np.log(np.maximum(Qs, 1e-300))),
                       np.interp(x, xs, Is)])
    else:
        Q_guess = np.asarray(guess_fn(x), dtype=float)
        if np.any(Q_guess <= 0):
            raise ValueError("guess_fn(x) must be strictly positive on [xmin, xmax].")
        # The ramp below starts at F = D = 0, so seed the charge there too: the
        # -2F x Qx^2 term evaluated at the target F contradicts the first solve,
        # increasingly so as F grows.
        ramping = homotopy and seed is None and (F != 0 or D != 0)
        I_guess = charge_from_flux(x, Q_guess, np.gradient(Q_guess, x), A,
                                   0.0 if ramping else D, 0.0 if ramping else F)
        Y = np.vstack([np.log(Q_guess), I_guess])

    m_out = background_mass(source_fn, Q_bg, x_out, A) if m_bg is None else float(m_bg)
    k = m_out + 1.0 / x_out

    def make_problem(Fk, Dk):
        def fun(xx, YY):
            u, Icharge = YY
            # clip before exp: u can swing far during Newton iterations, and an
            # overflow there poisons the residual and Jacobian.
            Q = np.exp(np.clip(u, -60.0, 60.0))
            Qx = charge_gradient(xx, Q, Icharge, A, Dk, Fk)
            return np.vstack([Qx / Q, -xx**2 * (S * delta_x(xx) + source_fn(xx, Q))])

        def bc(Ya, Yb):
            if outer_bc == "dirichlet":
                return np.array([Ya[1], Yb[0] - np.log(Q_bg)])
            Qb = np.exp(np.clip(Yb[0], -60.0, 60.0))
            Qxb = charge_gradient(np.array([x_out]), np.array([Qb]),
                                  np.array([Yb[1]]), A, Dk, Fk)[0]
            return np.array([Ya[1], Qxb + k * (Qb - Q_bg)])
        return fun, bc

    if homotopy and seed is None and (F != 0 or D != 0):
        F_path = np.linspace(0.0, F, steps + 1)
        D_path = np.linspace(0.0, D, steps + 1)
    else:
        F_path, D_path = [F], [D]

    x_cur, Y_cur = x, Y
    for Fk, Dk in zip(F_path, D_path):
        fun, bc = make_problem(Fk, Dk)
        sol = solve_bvp(fun, bc, x_cur, Y_cur, tol=tol, max_nodes=max_nodes, verbose=0)
        x_cur, Y_cur = sol.x, sol.y
        if not sol.success:
            print(f"[warn] charge-form (log) homotopy step F={Fk:.3g} failed: {sol.message}")
            break

    meta = dict(A=A, D=D, E=E, F=F, S=S,
                xmin=float(x[0]), xmax=x_out,
                delta_c=delta_c, delta_inf=delta_inf, eps_edge=eps_edge,
                Q_bg=Q_bg, k=k, m_bg=m_out, outer_bc=outer_bc,
                indep_var='x', formulation='charge-log')
    return LogChargeBVPResult(sol, A, D, F, meta)
