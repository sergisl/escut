"""Analytical utilities: density profile, scalar charge, initial guesses, n-slope computation, and thin-shell approximation."""
import numpy as np
from typing import Tuple
from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid


def _cross_at_one(x: np.ndarray, R: np.ndarray):
    """Estimate x where R(x)=1 using log–log interpolation.

    Parameters
    ----------
    x : ndarray
        Monotonic grid of independent variable.
    R : ndarray
        Positive ratio curve.

    Returns
    -------
    float or None
        Crossing location, or None if no sign-change around 1 is found.
    """
    x = np.asarray(x)
    R = np.asarray(R)
    m = np.isfinite(x) & np.isfinite(R) & (R > 0)
    xx, RR = x[m], R[m]
    if xx.size < 2:
        return None
    s = np.sign(RR - 1.0)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if idx.size == 0:
        return None
    j = idx[0]
    lx1, lx2 = np.log(xx[j]), np.log(xx[j + 1])
    ly1, ly2 = np.log(RR[j]), np.log(RR[j + 1])
    t = -ly1 / (ly2 - ly1 + 1e-300)
    return float(np.exp(lx1 + t * (lx2 - lx1)))

def smoothed_density(x: np.ndarray, delta_c: float, delta_inf: float = 0.0, 
                     eps: float = 1e-3) -> np.ndarray:
    """
    Smooth top-hat density profile.

    Transitions smoothly from delta_c (interior) to delta_inf (exterior) across the
    object boundary at x = 1. Used to avoid discontinuities that can cause numerical issues.

    Parameters
    ----------
    x : ndarray or float
        Coordinate(s) (x = r/R)
    delta_c : float
        Interior (central) overdensity
    delta_inf : float, optional
        Exterior (far-field) overdensity (default 0.0 for vacuum)
    eps : float, optional
        Transition width at object boundary (default 1e-3)

    Returns
    -------
    delta : ndarray or float
        Density profile at x
    """
    s = 0.5 * (1 - np.tanh((x - 1.0) / eps))
    return s * delta_c + (1 - s) * delta_inf

def mu_for_tanh_step(x: np.ndarray,
                     delta_c: float,
                     delta_inf: float = 0.0,
                     eps_edge: float = 1e-3,
                     S: float = 1.0) -> float:
    """
    Total scalar "charge" μ for a tanh-smoothed top-hat density on grid x.

    Definition (dimensionless variables):
        μ = (S/2) ∫ δ(x) x^2 dx,
    where δ(x) transitions from δ_c (interior) to δ_inf (exterior) with width eps_edge
    around x≈1 using a tanh smooth step.

    Parameters
    ----------
    x : ndarray
        1D grid of x = r/R where to perform the integral (monotonic increasing recommended)
    delta_c : float
        Interior overdensity value
    delta_inf : float, optional
        Exterior overdensity value (default 0)
    eps_edge : float, optional
        Transition width of the tanh edge (default 1e-3)
    S : float, optional
        Source coupling amplitude in the ODE (default 1)

    Returns
    -------
    mu : float
        Total μ integrated over the provided grid.
    """
    x = np.asarray(x, dtype=float)
    delta = smoothed_density(x, delta_c, delta_inf, eps_edge)
    integrand = delta * x**2
    # Simple trapezoidal integral over the provided grid
    mu = 0.5 * S * np.trapezoid(integrand, x)
    return float(mu)

def enclosed_source(x: np.ndarray, delta_c: float, delta_inf: float = 0.0,
                    eps_edge: float = 1e-3, S: float = 1.0) -> np.ndarray:
    """Enclosed scalar source I(x) = S ∫₀ˣ s² δ̂(s) ds (no ½ factor).

    This is the right-hand side of the integrated master equation
    (notes Eq. integrated_master), i.e. the quantity that sets both the
    Vainshtein gradient and the unscreened linear plateau x²y'_lin = I/A.
    The integral is evaluated on a fine internal log grid and interpolated to
    `x`, so its accuracy is independent of the (possibly coarse) plot grid.

    Parameters
    ----------
    x : ndarray            Radii where I(x) is wanted (x = r/R).
    delta_c, delta_inf : float  Interior / exterior density contrast.
    eps_edge : float       tanh transition width at x = 1.
    S : float              Source coupling amplitude.

    Returns
    -------
    I : ndarray  Enclosed source, same shape as x.
    """
    x = np.asarray(x, dtype=float)
    lo = max(float(np.min(x)), 1e-6)
    hi = float(np.max(x))
    xf = np.logspace(np.log10(lo), np.log10(hi), 6000)
    integrand = S * xf**2 * smoothed_density(xf, delta_c, delta_inf, eps_edge)
    If = cumulative_trapezoid(integrand, xf, initial=0.0)
    If += S * delta_c * xf[0]**3 / 3.0          # analytic [0, xf[0]] core tail
    return np.interp(x, xf, If)


def vainshtein_flux(x: np.ndarray, A: float, F: float, delta_c: float,
                    delta_inf: float = 0.0, eps_edge: float = 1e-3,
                    S: float = 1.0) -> np.ndarray:
    """Exact pure-Vainshtein gradient y'(x) from the first integral.

    The integrated master equation A x² y' + 2F x (y')² = I(x) is a quadratic
    in y'; selecting the branch that vanishes at infinity gives

        y'(x) = ( −A x + √(A²x² + 8F·I(x)/x) ) / (4F).

    For F = 0 this reduces smoothly to the linear gradient I/(A x²).  No solver
    is involved, so the result is exact and cannot fail to converge.

    Parameters
    ----------
    x : ndarray            Radii (x = r/R).
    A : float              Kinetic coefficient ε(1+6α²) (pure Vainshtein: ε).
    F : float              Vainshtein coefficient β̃.
    delta_c, delta_inf, eps_edge, S : float  Density / source parameters.

    Returns
    -------
    yp : ndarray  Gradient y'(x), same shape as x.
    """
    x = np.asarray(x, dtype=float)
    I = enclosed_source(x, delta_c, delta_inf, eps_edge, S)
    if F == 0:
        return I / (A * x**2)
    disc = A**2 * x**2 + 8.0 * F * I / x
    return (-A * x + np.sqrt(np.maximum(disc, 0.0))) / (4.0 * F)


def vainshtein_radius(A: float, F: float, delta_c: float, delta_inf: float = 0.0,
                      eps_edge: float = 1e-3, S: float = 1.0) -> float:
    """Pure-Vainshtein radius x_V = (8 F I_obj / A²)^(1/3).

    The transition from the Vainshtein-screened (slope-3/2) regime to the
    unscreened linear plateau occurs where the two terms of the integrated
    master equation balance, A²x² = 8F I/x, with the *object's* enclosed
    overdensity I_obj = S ∫ s²(δ̂ − δ_inf) ds (the background is subtracted so
    the integral converges and the radius is that of the localised structure).

    Returns NaN if F = 0 (no Vainshtein term).
    """
    if F == 0:
        return float("nan")
    xs = np.logspace(-3, 2, 4000)          # overdensity → 0 well before x ~ 100
    integrand = S * xs**2 * (smoothed_density(xs, delta_c, delta_inf, eps_edge) - delta_inf)
    I_obj = np.trapezoid(integrand, xs) + S * (delta_c - delta_inf) * xs[0]**3 / 3.0
    return float((8.0 * F * I_obj / A**2)**(1.0 / 3.0))


class _AlgebraicVainshteinSolution:
    """Lightweight stand-in for a scipy BVPResult holding the *exact* Vainshtein flux.

    Exposes the same interface `plot_double_panel` consumes — a callable `.sol(x)`
    returning a (2, N) array whose row [1] is y'(x), plus a `.meta` dict — so the
    algebraic solution can be dropped into the existing plotting path with no
    second-order BVP solve.  Row [0] (the field itself) is left as NaN: only the
    flux x²|y'| is used by the figure.
    """

    def __init__(self, A: float, F: float, delta_c: float,
                 delta_inf: float = 0.0, eps_edge: float = 1e-3, S: float = 1.0):
        self._kw = dict(A=A, F=F, delta_c=delta_c, delta_inf=delta_inf,
                        eps_edge=eps_edge, S=S)
        self.success = True
        self.meta = dict(A=A, F=F, delta_c=delta_c, delta_inf=delta_inf,
                         eps_edge=eps_edge, S=S, indep_var='x', exact=True)

    def sol(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        yp = vainshtein_flux(x, **self._kw)
        return np.vstack([np.full_like(x, np.nan), yp])


def make_vainshtein_solution(A: float, F: float, delta_c: float,
                             delta_inf: float = 0.0, eps_edge: float = 1e-3,
                             S: float = 1.0) -> _AlgebraicVainshteinSolution:
    """Build an exact pure-Vainshtein 'solution' for `plot_double_panel`.

    Pass F = 0.0 to obtain the matching linear (unscreened) reference.
    """
    return _AlgebraicVainshteinSolution(A, F, delta_c, delta_inf, eps_edge, S)


def Q_small_r(B: float, C: float, S: float, delta_c: float, G: float = 0) -> float:
    """Algebraic interior solution Q_in from G·Q³ + C·Q² + B·Q + S·δ_c = 0 (r << R limit).

    For G=0 uses the analytic quadratic formula (screened branch).
    For G≠0 uses np.roots and selects the real root closest to the linear estimate -S·δ_c/B.

    Parameters
    ----------
    B, C : float     Linear and quadratic mass coefficients.
    S : float        Source coupling amplitude.
    delta_c : float  Central overdensity.
    G : float        Cubic self-interaction coefficient (default 0).

    Returns
    -------
    Q_in : float  Interior field value.

    Raises
    ------
    ValueError  If no valid root is found.
    """
    if G == 0:
        if C == 0:
            # Linear limit B*Q + S*delta_c = 0; the quadratic branch below is
            # 0/0 here. With no mass term either there is no interior value.
            return -S * delta_c / B if B != 0 else float("nan")
        disc = B**2 - 4.0*C*S*delta_c
        if np.any(disc < 0):
            raise ValueError(
                f"Negative discriminant in Q_small_r: disc = {disc}. "
                f"Parameters (B={B}, C={C}, S={S}, delta_c={delta_c}) are not compatible. "
                f"Check: B^2 >= 4*C*S*delta_c"
            )
        return (-B - np.sqrt(disc)) / (2.0*C)
    else:
        roots = np.roots([G, C, B, S * delta_c])
        real_roots = roots[np.abs(roots.imag) < 1e-8 * (np.abs(roots.real) + 1e-10)].real
        if len(real_roots) == 0:
            raise ValueError(
                f"No real roots in Q_small_r cubic (G={G}, C={C}, B={B}, S={S}, delta_c={delta_c})."
            )
        Q_lin = -S * delta_c / B if B != 0 else 0.0
        return float(real_roots[np.argmin(np.abs(real_roots - Q_lin))])

def _mu_eff_Q_in_m_eff(A: float, B: float, C: float, S: float,
                       delta_c: float, R: float, G: float = 0) -> Tuple[float, float, float]:
    """Rough analytic estimate of (mu_eff, Q_in, m_eff) for make_initial_guess_from_analytic."""
    # Interior value
    Q_in = Q_small_r(B=B, C=C, S=S, delta_c=delta_c, G=G)

    # Effective mass at Q_in: m² = (B + 2C·Q_in + 3G·Q_in²) / A
    m2 = max(0.0, (B + 2.0*C*Q_in + 3.0*G*Q_in**2) / A) if A != 0 else 0.0
    m_eff = np.sqrt(m2)
    
    # Rough estimate of mu_eff from flux continuity
    # This assumes Q_out ~ (mu_eff / (8π A r)) * exp(-m_eff * r)
    # At r = R, matching Q and its derivative gives:
    #   Q_in ≈ (mu_eff / (8π A R)) * exp(-m_eff * R) / (1 + m_eff * R)
    # Solving for mu_eff:
    rho_m = 1.0  # reference density (or use appropriate value)
    
    mu_eff = ((4.0 * np.pi * A * R * Q_in * (1.0 + m_eff * R) * rho_m) / S)
    
    return mu_eff, Q_in, m_eff

def make_initial_guess_from_analytic(x: np.ndarray, A: float, B: float, C: float,
                                     S: float, delta_c: float, R: float,
                                     delta_R: float = 0.1, G: float = 0) -> np.ndarray:
    """Build a smooth [Q, Qx] initial guess: constant Q_in interior, Yukawa exterior, tanh blend.

    Parameters
    ----------
    x : ndarray          Coordinate grid (x = r/R).
    A, B, C, S : float   Equation coefficients.
    delta_c : float      Central overdensity.
    R : float            Object radius in x-units (typically 1.0).
    delta_R : float      tanh blend half-width (default 0.1).
    G : float            Cubic self-interaction coefficient (default 0).

    Returns
    -------
    y_guess : ndarray, shape (2, N)  Initial guess [Q(x), Qx(x)].

    Raises
    ------
    ValueError  If Q_small_r finds no valid root.
    """
    # Compute interior value
    Q_in = Q_small_r(B=B, C=C, S=S, delta_c=delta_c, G=G)

    # Compute exterior parameters
    mu_eff, _, m_eff = _mu_eff_Q_in_m_eff(A, B, C, S, delta_c, R, G=G)
    
    # Avoid division by zero at x=0
    safe_x = np.where(x == 0.0, 1e-10, x)
    
    # Exterior Yukawa-like solution
    # Q_out ~ (mu_eff / (8π A r)) * exp(-m_eff * r) with normalization
    Q_out = (mu_eff / (8.0 * np.pi * A * safe_x)) * np.exp(-m_eff * (x - R)) / (1.0 + m_eff * R)
    
    # Smooth interpolation between interior and exterior
    # tanh provides C-infinity smoothness
    # Centered at x = R with width delta_R
    smooth_factor = 0.5 * (1.0 + np.tanh((x - R) / delta_R))
    Q = Q_in * (1.0 - smooth_factor) + Q_out * smooth_factor
    
    # Approximate derivative using finite differences
    # This is adequate for an initial guess
    Qx = np.gradient(Q, x)
    
    return np.vstack([Q, Qx])

def analytic_Q_thin_shell(x_eval, *, A: float, B: float, C: float, S: float,
                          delta_c: float, delta_inf: float, eps_edge: float,
                          R: float = 1.0, shell_frac: float = 0.10,
                          G: float = 0) -> np.ndarray:
    """
    Thin-shell analytical profile for the chameleon scalar field.

    Exterior (x >= R): Q_ext(x) ~ [μ_eff / (4π A x)] · exp[−m_eff (x−R)] / (1 + m_eff R)
    Interior (x < R):  Q_in    ~ Q_small_r(B, C, S, delta_c, G)

    μ_eff is the integrated source from the shell layer x ∈ [R−ΔR, R].

    Parameters
    ----------
    x_eval : array_like  Evaluation grid.
    A, B, C, S : float   Equation coefficients.
    delta_c, delta_inf, eps_edge : float  Density profile parameters.
    R : float            Object radius in code units (default 1).
    shell_frac : float   Thin-shell fractional thickness ΔR/R (default 0.1).
    G : float            Cubic self-interaction coefficient (default 0).

    Returns
    -------
    ndarray  |Q(x)| thin-shell approximation, same shape as x_eval.
    """
    x_eval = np.asarray(x_eval, dtype=float)
    Q_in  = Q_small_r(B=B, C=C, S=S, delta_c=delta_c, G=G)
    m2    = max(0.0, (B + 2.0*C*Q_in + 3.0*G*Q_in**2) / A) if A != 0 else 0.0
    m_eff = np.sqrt(m2)
    xL    = max(float(np.min(x_eval)), R * (1.0 - shell_frac))
    xs    = np.linspace(xL, R, 2500)
    mu_eff = 0.5 * S * np.trapezoid(
        smoothed_density(xs, delta_c, delta_inf, eps=eps_edge) * xs**2, xs
    )
    safe_x = np.where(x_eval <= 0.0, 1e-12, x_eval)
    if m_eff > 0.0:
        Q_ext = (mu_eff / (4.0 * np.pi * A * safe_x)) * np.exp(-m_eff * (safe_x - R)) / (1.0 + m_eff * R)
    else:
        Q_ext = mu_eff / (4.0 * np.pi * A * safe_x)
    return np.abs(np.where(x_eval >= R, Q_ext, Q_in))


def compute_n_slope(sol, x: np.ndarray, win: int = 61, poly: int = 3, eps: float = 1e-300) -> np.ndarray:
    """
    Compute n(x) = d ln[x^2 |Q_x|] / d ln x on a given grid x.

    Parameters
    ----------
    sol : scipy BVPResult
        Solution object with .sol callable returning [Q, Qx].
    x : ndarray
        1D grid of x = r/R (prefer log-spaced to get uniform ln x spacing).
    win : int, optional
        Savitzky-Golay smoothing window (must be odd, >= 5). If < 5, smoothing is skipped.
    poly : int, optional
        Savitzky-Golay polynomial order (default 3).
    eps : float, optional
        Small regulariser inside the log to avoid -inf when Qx ~ 0.

    Returns
    -------
    n : ndarray
        Local power-law slope n(x), same shape as x.
    """
    Qx = sol.sol(x)[1]
    L = np.log(x**2 * np.abs(Qx) + eps)
    if win % 2 == 0:
        win += 1
    Ls = savgol_filter(L, win, poly) if win >= 5 else L
    n = np.gradient(Ls, np.log(x))
    return n


def V_eff(Q: np.ndarray, A: float, B: float, C: float,
          S: float, delta_c: float, G: float = 0) -> np.ndarray:
    """Effective potential V_eff(Q; δ_c) from the homogeneous scalar EOM.

    The EOM ∇²Q = (S·δ_c + B·Q + C·Q² + G·Q³) / A  ≡  −dV_eff/dQ  implies

        V_eff(Q; δ_c) = −(S·δ_c·Q + B·Q²/2 + C·Q³/3 + G·Q⁴/4) / A

    Parameters
    ----------
    Q : array_like
        Field values at which to evaluate V_eff.
    A : float
        Kinetic coefficient A in the EOM.
    B : float
        Linear mass coefficient B.
    C : float
        Quadratic self-interaction coefficient C.
    S : float
        Source coupling S.
    delta_c : float
        Density contrast δ_c (sets the depth of the potential well).
    G : float
        Cubic self-interaction coefficient (default 0).

    Returns
    -------
    V : ndarray
        Effective potential values, same shape as Q.
    """
    Q = np.asarray(Q, dtype=float)
    return -(S * delta_c * Q + (B / 2.0) * Q**2 + (C / 3.0) * Q**3 + (G / 4.0) * Q**4) / A


# ---------------------------------------------------------------------------
# Jordan-frame metric sector: dynamical potential and effective Newton constant
# ---------------------------------------------------------------------------
#
# The scalar flux Q'(x) is not by itself a field-redefinition-invariant
# observable; the physical quantity that governs matter geodesics is the
# Jordan-frame dynamical potential Phi (metric ds^2 = -(1+2Phi)dt^2 +
# a^2(1-2Psi)dx^2, so non-relativistic geodesics feel Phi).  Substituting the
# screened scalar profile back into the metric equations of the paper,
#
#     (0,0):        2 lap(Psi) + alpha_B lap(Q) = a^2 rho_m delta
#     traceless:    lap(Psi - Phi - alpha_M Q) = 0,
#
# eliminating Psi and integrating once over a sphere (using
# int_0^r lap(f) s^2 ds = r^2 f'(r)) gives the total physical acceleration as
# the Newtonian piece plus a scalar fifth force,
#
#     Phi'(r) = Phi'_GR(r) - gamma Q'(r),   gamma = alpha_B/2 + alpha_M,
#     Phi'_GR(r) = mu(r)/r^2   (enclosed-mass Newtonian acceleration).
#
# In the linear regime Q' -> Q'_lin = -(alpha_B+2alpha_M) mu / (Gamma r^2), so
#
#     G_eff/G_N = Phi'/Phi'_GR
#               = 1 + (alpha_B + 2 alpha_M)^2 / (2 Gamma) * [Q'(x)/Q'_lin(x)],
#
# which saturates to the unscreened enhancement 1 + (alpha_B+2alpha_M)^2/(2Gamma)
# far from the source and returns to 1 (GR) wherever Q' is screened.  The
# prefactor vanishes iff alpha_B + 2 alpha_M = 0 -- the same combination that
# decouples the scalar from matter in the master equation.


def screening_factor(Qx: np.ndarray, Qx_lin: np.ndarray) -> np.ndarray:
    """Scalar screening factor S(x) = Q'(x) / Q'_lin(x).

    S -> 1 is unscreened (linear), S -> 0 is fully screened.
    """
    Qx = np.asarray(Qx, dtype=float)
    Qx_lin = np.asarray(Qx_lin, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return Qx / Qx_lin


def effective_gravity(Qx: np.ndarray, Qx_lin: np.ndarray,
                      alpha_B: float, alpha_M: float, Gamma: float = 1.0) -> dict:
    """Jordan-frame effective Newton constant G_eff/G_N = Phi'/Phi'_GR.

    Obtained by substituting the scalar flux back into the metric equations
    (see the module notes above):

        G_eff/G_N = 1 + (alpha_B + 2 alpha_M)^2 / (2 Gamma) * [Q'(x)/Q'_lin(x)].

    The ratio -> 1 + (alpha_B+2alpha_M)^2/(2 Gamma) where the field is
    unscreened, and -> 1 (GR) where it is screened.  No free normalisation: the
    unscreened enhancement is fixed by the EFT functions and the kinetic term.

    Parameters
    ----------
    Qx : ndarray      Screened (nonlinear) scalar flux Q'(x) = dQ/dx.
    Qx_lin : ndarray  Linear reference flux Q'_lin(x) (nonlinear terms off),
                      on the same grid as Qx.
    alpha_B : float   Braiding parameter (kinetic mixing of Q into the (0,0) eq).
    alpha_M : float   Running / anisotropic-stress parameter (slip).
    Gamma : float     Linear kinetic coefficient Gamma = D c_s^2 (the solver's
                      `A`).  Default 1.

    Returns
    -------
    dict with keys
        'Xi'            : unscreened enhancement (alpha_B+2alpha_M)^2/(2 Gamma).
        'plateau'       : 1 + Xi (the unscreened value of G_eff/G_N).
        'screening'     : screening factor S(x) = Q'/Q'_lin.
        'Geff_over_GN'  : effective coupling ratio 1 + Xi S(x).
    """
    Xi = (alpha_B + 2.0 * alpha_M) ** 2 / (2.0 * Gamma)
    S = screening_factor(Qx, Qx_lin)
    return dict(Xi=Xi, plateau=1.0 + Xi, screening=S, Geff_over_GN=1.0 + Xi * S)