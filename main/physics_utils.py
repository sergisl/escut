"""Analytical utilities: density profile, scalar charge, initial guesses, n-slope computation, and thin-shell approximation."""
import numpy as np
from typing import Tuple
from scipy.signal import savgol_filter


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

def Q_small_r(B: float, C: float, S: float, delta_c: float) -> float:
    """Algebraic interior solution Q_in from C·Q² + B·Q + S·δ_c = 0 (r << R limit).

    Takes the screened branch Q_in = (-B - sqrt(B²-4CSδ_c)) / (2C).

    Parameters
    ----------
    B, C : float  Linear and nonlinear mass coefficients.
    S : float     Source coupling amplitude.
    delta_c : float  Central overdensity.

    Returns
    -------
    Q_in : float  Interior field value.

    Raises
    ------
    ValueError  If discriminant B²-4CSδ_c < 0.
    """
    disc = B**2 - 4.0*C*S*delta_c
    if np.any(disc < 0):
        raise ValueError(
            f"Negative discriminant in Q_small_r: disc = {disc}. "
            f"Parameters (B={B}, C={C}, S={S}, delta_c={delta_c}) are not compatible. "
            f"Check: B^2 >= 4*C*S*delta_c"
        )
    # Choose the branch appropriate for the screened solution
    # For C > 0 (chameleon-like), take the smaller magnitude branch: (-B - sqrt(disc))/(2C)
    Q_in = (-B - np.sqrt(disc)) / (2.0*C)
    return Q_in

def _mu_eff_Q_in_m_eff(A: float, B: float, C: float, S: float,
                       delta_c: float, R: float) -> Tuple[float, float, float]:
    """Rough analytic estimate of (mu_eff, Q_in, m_eff) for make_initial_guess_from_analytic."""
    # Interior value
    Q_in = Q_small_r(B=B, C=C, S=S, delta_c=delta_c)
    
    # Effective mass (ensure non-negative for real exponent)
    m2 = max(0.0, -B / A) if A != 0 else 0.0
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
                                     delta_R: float = 0.1) -> np.ndarray:
    """Build a smooth [Q, Qx] initial guess: constant Q_in interior, Yukawa exterior, tanh blend.

    Parameters
    ----------
    x : ndarray       Coordinate grid (x = r/R).
    A, B, C, S : float  Equation coefficients.
    delta_c : float   Central overdensity.
    R : float         Object radius in x-units (typically 1.0).
    delta_R : float   tanh blend half-width (default 0.1).

    Returns
    -------
    y_guess : ndarray, shape (2, N)  Initial guess [Q(x), Qx(x)].

    Raises
    ------
    ValueError  If Q_small_r discriminant is negative.
    """
    # Compute interior value
    Q_in = Q_small_r(B=B, C=C, S=S, delta_c=delta_c)
    
    # Compute exterior parameters
    mu_eff, _, m_eff = _mu_eff_Q_in_m_eff(A, B, C, S, delta_c, R)
    
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
                          R: float = 1.0, shell_frac: float = 0.10) -> np.ndarray:
    """
    Thin-shell analytical profile for the chameleon scalar field.

    Exterior (x >= R): Q_ext(x) ~ [μ_eff / (4π A x)] · exp[−m_eff (x−R)] / (1 + m_eff R)
    Interior (x < R):  Q_in    ~ Q_small_r(B, C, S, delta_c)

    μ_eff is the integrated source from the shell layer x ∈ [R−ΔR, R].

    Parameters
    ----------
    x_eval : array_like  Evaluation grid.
    A, B, C, S : float   Equation coefficients.
    delta_c, delta_inf, eps_edge : float  Density profile parameters.
    R : float            Object radius in code units (default 1).
    shell_frac : float   Thin-shell fractional thickness ΔR/R (default 0.1).

    Returns
    -------
    ndarray  |Q(x)| thin-shell approximation, same shape as x_eval.
    """
    x_eval = np.asarray(x_eval, dtype=float)
    m2    = max(0.0, -B / A) if A != 0 else 0.0
    m_eff = np.sqrt(m2)
    Q_in  = Q_small_r(B=B, C=C, S=S, delta_c=delta_c)
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
          S: float, delta_c: float) -> np.ndarray:
    """Effective potential V_eff(Q; δ_c) from the homogeneous scalar EOM.

    The EOM ∇²Q = (S·δ_c + B·Q + C·Q²) / A  ≡  −dV_eff/dQ  implies

        V_eff(Q; δ_c) = −(S·δ_c·Q + B·Q²/2 + C·Q³/3) / A

    Parameters
    ----------
    Q : array_like
        Field values at which to evaluate V_eff.
    A : float
        Kinetic coefficient A in the EOM.
    B : float
        Linear mass coefficient B.
    C : float
        Cubic self-interaction coefficient C.
    S : float
        Source coupling S.
    delta_c : float
        Density contrast δ_c (sets the depth of the potential well).

    Returns
    -------
    V : ndarray
        Effective potential values, same shape as Q.
    """
    Q = np.asarray(Q, dtype=float)
    return -(S * delta_c * Q + (B / 2.0) * Q**2 + (C / 3.0) * Q**3) / A