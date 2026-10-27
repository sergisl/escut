"""Mixed Vainshtein + Chameleon screening: model set-up and diagnostics, built
on escut's charge formulation.

The master equation is solved for the *full* scalar y = phi/phi_* rather than a
perturbation, so this uses escut's full-field mode (`source_fn`):

    A (x^2 y')'/x^2 + 2 beta [x (y')^2]'/x^2 = rho(x) e^{-xi y} - chi / y^2,

with A = eps * (1 + 6 alpha^2) the kinetic coefficient, beta the Vainshtein
strength, xi the conformal coupling and chi toggling the Chameleon potential.

Sign convention: escut's charge has the Vainshtein term entering as -2F x Qx^2,
the opposite of the convention above, so beta is mapped to escut's F = -beta
inside `solve`. Work in `beta` throughout this module; do not pass F directly.
"""

import os
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAIN = os.path.join(_HERE, "..", "..", "main")
if _MAIN not in sys.path:
    sys.path.insert(0, _MAIN)

from charge_solver import (solve_scalar_bvp_charge_log, charge_gradient,
                           charge_vainshtein_radius)
from physics_utils import smoothed_density


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def y_minimum(rho, xi=0.0, chi=1.0):
    """Field value minimising the effective potential at density `rho`.

    Root of rho e^{-xi y} - chi / y^2 = 0. For xi = 0 this is the closed form
    y = sqrt(chi / rho). For xi > 0 the source has two roots or none: writing
    the condition as g(y) = rho y^2 e^{-xi y} / chi = 1, g peaks at y = 2/xi,
    so a minimum exists only while

        xi <= 2 sqrt(rho / chi) / e,

    beyond which it annihilates with the neighbouring maximum (NaN is returned).
    Where two roots exist the stable one is the smaller.
    """
    if rho <= 0:
        return np.inf
    if xi == 0.0:
        return np.sqrt(chi / rho)
    y_peak = 2.0 / xi
    if rho * y_peak**2 * np.exp(-2.0) / chi < 1.0:
        return np.nan
    f = lambda y: rho * np.exp(-xi * y) - chi / y**2
    return brentq(f, min(np.sqrt(chi / rho), y_peak) * 1e-4, y_peak, maxiter=200)


def density(x, rho_in, rho_bg, eps_edge=0.015):
    """Smoothed top-hat density: rho_in inside x = 1, rho_bg outside."""
    return smoothed_density(x, rho_in, rho_bg, eps=eps_edge)


def density_profile_fn(rho_in, rho_bg, eps_edge=0.015):
    """The default smoothed top hat, as a callable (for use as `dens_fn`)."""
    return lambda x: density(x, rho_in, rho_bg, eps_edge)


def nfw_profile(rho_in, rho_bg, r_s=0.15, r_c=0.5, x_trunc=6.0, w=0.8, x_ref=1e-2):
    """Cuspy NFW-like halo, regularised so the centre is finite.

    A true NFW diverges as 1/r at the centre; the enclosed mass and hence the
    field stay regular, but a solver needs a finite value, so an inner core is
    added. Truncated outside `x_trunc` to keep the total mass finite.
    """
    trunc = lambda x: 0.5 * (1 - np.tanh((x - x_trunc) / w))
    raw = lambda x: trunc(x) / ((1 + x / r_s) * (1 + x / r_c)**2)
    norm = raw(np.full(1, x_ref))[0]
    return lambda x: rho_bg + (rho_in - rho_bg) * raw(np.asarray(x, float)) / norm


def burkert_profile(rho_in, rho_bg, r_s=0.6, x_trunc=6.0, w=0.8, x_ref=1e-2):
    """Cored Burkert halo, the shape that better fits dwarf-galaxy rotation
    curves. Flat centre, so the field has something to pin to."""
    trunc = lambda x: 0.5 * (1 - np.tanh((x - x_trunc) / w))
    raw = lambda x: trunc(x) / ((1 + x / r_s) * (1 + (x / r_s)**2))
    norm = raw(np.full(1, x_ref))[0]
    return lambda x: rho_bg + (rho_in - rho_bg) * raw(np.asarray(x, float)) / norm


def compensated_void_profile(rho_in, rho_bg, r_ridge=1.2, p=5):
    """Void with an overdense ridge at its edge, as seen in simulations.

    The enclosed charge is non-monotonic and changes sign, which is a stricter
    test of the resolution than a monotonic top hat.
    """
    delta = rho_in / rho_bg - 1.0
    return lambda x: rho_bg * (1 + delta * (1 - (np.asarray(x, float))**2)
                               / (1 + (np.asarray(x, float) / r_ridge)**p))


def make_source(rho_in, rho_bg, eps_edge=0.015, xi=0.0, chi=1.0, dens_fn=None):
    """`source_fn(x, y)` for escut, i.e. minus the right-hand side above."""
    rho = dens_fn if dens_fn is not None else (lambda xx: density(xx, rho_in, rho_bg, eps_edge))
    return lambda x, y: chi / y**2 - rho(x) * np.exp(-xi * y)


def make_guess(rho_in, rho_bg, eps_edge=0.015, xi=0.0, chi=1.0, dens_fn=None):
    """`guess_fn(x)` tracking the local potential minimum y_v(rho(x))."""
    rho = dens_fn if dens_fn is not None else (lambda xx: density(xx, rho_in, rho_bg, eps_edge))
    def guess(x):
        return np.array([y_minimum(d, xi, chi) for d in np.atleast_1d(rho(x))])
    return guess


def compton_wavelength(A, rho_bg, xi=0.0, chi=1.0):
    """Background Compton wavelength, 1/m with m^2 = (2 chi / y_bg^3) / A."""
    y_bg = y_minimum(rho_bg, xi, chi)
    m2 = (2.0 * chi / y_bg**3 - xi * chi / y_bg**2) / A
    return np.inf if m2 <= 0 else 1.0 / np.sqrt(m2)


def default_x_max(A, beta, rho_in, rho_bg, xi=0.0, chi=1.0, pad=12.0):
    """Outer domain comfortably outside both the Vainshtein sphere and the
    background Compton wavelength.

    Sizing on the Vainshtein radius alone leaves the outer boundary inside the
    Chameleon range when the background is dilute, where the Robin condition is
    then imposed before the field has reached its asymptote.
    """
    I_tot = max(rho_in - rho_bg, 0.0) / 3.0
    xV = (8.0 * beta * I_tot / A**2) ** (1.0 / 3.0) if beta > 0 else 1.0
    lam = compton_wavelength(A, rho_bg, xi, chi)
    scales = [50.0, pad * xV] + ([pad * lam] if np.isfinite(lam) else [])
    return float(min(max(scales), 5e4))


def shell_mesh(x_min, x_max, n_log=2500, n_shell=1500, shell=None):
    """Log grid with extra nodes clustered on the screening shell near x = 1.

    The clustered window defaults to (0.5, min(15, x_max/2)), wide enough to
    cover the Vainshtein transition, which moves well outside x = 1 at large
    beta. Pass `shell=(lo, hi)` to set it explicitly.
    """
    if shell is None:
        shell = (0.5, min(15.0, x_max / 2.0))
    lo, hi = max(shell[0], x_min), min(shell[1], x_max)
    parts = [np.logspace(np.log10(x_min), np.log10(x_max), n_log)]
    if hi > lo:
        parts.append(np.linspace(lo, hi, n_shell))
    return np.unique(np.concatenate(parts))


def solve(*, A, beta, rho_in, rho_bg, eps_edge=0.015, xi=0.0, chi=1.0,
          dens_fn=None, x_min=1e-2, x_max=None, x_init=None,
          n_log=2500, n_shell=1500, tol=1e-7, max_nodes=2_000_000,
          steps=6, seed=None):
    """Solve the mixed-screening equation for y(x).

    Returns a `LogChargeBVPResult`; `.sol(x)` gives [y, y'] and
    `.discriminant(x)` the escut-convention discriminant (see
    `discriminant_ratio` for the normalised form used in the paper).

    Parameters
    ----------
    A : float        Kinetic coefficient eps * (1 + 6 alpha^2).
    beta : float     Vainshtein strength (mapped to escut's F = -beta).
    rho_in, rho_bg : float  Interior and background density.
    eps_edge : float        Width of the density transition at x = 1.
    xi, chi : float         Conformal coupling; Chameleon potential on/off.
    dens_fn : callable, optional  Custom density profile rho(x).
    x_max : float, optional       Defaults to `default_x_max`.
    x_init : ndarray, optional    Mesh, overriding `shell_mesh`.
    seed : (x, y, I), optional    Warm start from a previous solution.
    """
    y_bg = y_minimum(rho_bg, xi, chi)
    if not np.isfinite(y_bg):
        raise ValueError(f"no background minimum for rho_bg={rho_bg}, xi={xi} "
                         "(the potential minimum has annihilated; reduce xi)")
    if x_max is None:
        x_max = default_x_max(A, beta, rho_in, rho_bg, xi, chi)
    if x_init is None:
        x_init = shell_mesh(x_min, x_max, n_log, n_shell)

    return solve_scalar_bvp_charge_log(
        A=A, F=-beta, S=0.0,
        delta_c=rho_in, delta_inf=rho_bg, eps_edge=eps_edge,
        Q_bg=y_bg,
        source_fn=make_source(rho_in, rho_bg, eps_edge, xi, chi, dens_fn),
        guess_fn=make_guess(rho_in, rho_bg, eps_edge, xi, chi, dens_fn),
        x_init=x_init, tol=tol, max_nodes=max_nodes, steps=steps, seed=seed)


# ---------------------------------------------------------------------------
# Charges
# ---------------------------------------------------------------------------

def enclosed_charge(x, integrand_fn, n=6000):
    """int_0^x s^2 g(s) ds, evaluated on a fine log grid and interpolated."""
    x = np.asarray(x, dtype=float)
    xf = np.logspace(np.log10(max(x.min(), 1e-6)), np.log10(x.max()), n)
    g = integrand_fn(xf)
    If = cumulative_trapezoid(xf**2 * g, xf, initial=0.0) + xf[0]**3 * g[0] / 3.0
    return np.interp(x, xf, If)


def mass_charge(x, rho_in, rho_bg, eps_edge=0.015, dens_fn=None):
    """Charge sourced by the density perturbation alone, as in a pure Galileon
    with no potential to screen it."""
    rho = dens_fn if dens_fn is not None else (lambda xx: density(xx, rho_in, rho_bg, eps_edge))
    return enclosed_charge(x, lambda s: rho(s) - rho_bg)


def _check(sol):
    """Guard against reading a non-converged solve.

    scipy returns its last iterate whether or not it converged, so an
    unconverged profile yields plausible-looking but meaningless numbers.
    """
    if not getattr(sol, "success", True):
        raise ValueError(
            "solution did not converge, so diagnostics would be meaningless: "
            f"{getattr(sol, 'message', 'unknown reason')}. Re-solve (a larger "
            "max_nodes, more homotopy steps, or a mesh clustered on the "
            "screening shell usually fixes it) before measuring."
        )


def solved_charge(sol, x):
    """Enclosed charge I(x) of a solved profile."""
    _check(sol)
    return sol._interp(np.asarray(x, dtype=float))[1]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def discriminant_ratio(x, I, A, beta):
    """Normalised discriminant Delta / (A x)^2 = 1 + 8 beta I / (A^2 x^3).

    Negative values mean no real gradient exists: the imaginary fifth force of
    a pure Galileon in a sufficiently deep void.
    """
    x = np.asarray(x, dtype=float)
    return 1.0 + 8.0 * beta * np.asarray(I, dtype=float) / (A**2 * x**3)


def discriminant_ratio_from_gradient(x, yp, A, beta, x_floor=2e-3, plateau_below=0.05):
    """Normalised discriminant via the perfect square, Delta = (A x + 4 beta y')^2,

        Delta / (A x)^2 = (1 + 4 beta y' / (A x))^2.

    Algebraically identical to `discriminant_ratio` but numerically better near
    the origin, where the charge route divides a vanishing I by x^3. The ratio
    tends to a finite plateau as x -> 0, so values inside `plateau_below` are
    held at the first reliable one.
    """
    x = np.asarray(x, dtype=float)
    r = (1.0 + 4.0 * beta * np.asarray(yp, dtype=float) / (A * np.maximum(x, x_floor)))**2
    i0 = int(np.searchsorted(x, plateau_below))
    if 0 < i0 < r.size:
        r[:i0] = r[i0]
    return r


def gradient(x, I, A, beta):
    """Field gradient y'(x) implied by a charge I(x)."""
    x = np.asarray(x, dtype=float)
    return charge_gradient(x, np.zeros_like(x), np.asarray(I, dtype=float), A, 0.0, -beta)


def vainshtein_radius(x, I, A, beta):
    """Radius where the Vainshtein term overtakes the linear one."""
    return charge_vainshtein_radius(np.asarray(x, dtype=float),
                                    np.asarray(I, dtype=float), A, beta)


def vainshtein_radius_analytic(A, beta, rho_in, rho_bg):
    """Closed-form Vainshtein radius for an unscreened top hat,

        x_V = (8 beta I_tot / A^2)^(1/3),   I_tot = (rho_in - rho_bg) / 3,

    i.e. the whole mass perturbation sources the field. Use `A = eps` (not
    eps*at) for the shift-symmetric pure-Galileon baseline. Preferred over
    root-finding `vainshtein_radius` on a coarse grid, where interpolating a
    cubic costs a few tenths of a percent.
    """
    if beta <= 0:
        return float("nan")
    I_tot = max(rho_in - rho_bg, 0.0) / 3.0
    return float((8.0 * beta * I_tot / A**2) ** (1.0 / 3.0))


def radius_reduction(sol, x, A, beta, rho_in, rho_bg, eps_edge=0.015, dens_fn=None):
    """Mixed vs pure-Galileon Vainshtein radii and their ratio.

    The pure-Galileon comparison uses the shift-symmetric case (no conformal
    coupling, so its kinetic coefficient is eps rather than A) sourced by the
    full mass perturbation.
    """
    x = np.asarray(x, dtype=float)
    x_mixed = vainshtein_radius(x, solved_charge(sol, x), A, beta)
    x_pure = vainshtein_radius(x, mass_charge(x, rho_in, rho_bg, eps_edge, dens_fn), A, beta)
    return dict(x_mixed=x_mixed, x_pure=x_pure,
                ratio=x_mixed / x_pure if np.isfinite(x_pure) and x_pure > 0 else np.nan)


def descreening(sol, rho_in, rho_bg, xi=0.0, chi=1.0, x_core=1e-2):
    """Displacement of the central field from its interior minimum,

        D = (y_0 - y_in) / (y_bg - y_in),

    so D = 0 is a fully screened core and D -> 1 a de-screened one.
    """
    _check(sol)
    y_in = y_minimum(rho_in, xi, chi)
    y_bg = y_minimum(rho_bg, xi, chi)
    y0 = float(np.ravel(sol.sol(np.atleast_1d(x_core)))[0])
    span = y_bg - y_in
    return np.nan if span == 0 else (y0 - y_in) / span


def core_radius(sol, rho_in, xi=0.0, chi=1.0, tol=1.2, x_lo=1e-2, x_hi=2.0, n=800):
    """Edge of the screened Chameleon core: the outermost radius at which the
    field is still pinned within `tol` of its interior minimum y_in.

    Returns NaN when the centre itself is not pinned, i.e. no core has formed.
    Searched only inside x_hi, since a core edge outside the source would be
    unphysical.
    """
    _check(sol)
    y_in = y_minimum(rho_in, xi, chi)
    x = np.logspace(np.log10(x_lo), np.log10(x_hi), n)
    y = sol.sol(x)[0]
    pinned = y <= tol * y_in
    centred = y[np.argmin(np.abs(x - 2 * x_lo))] <= tol * y_in
    return float(x[pinned].max()) if (pinned.any() and centred) else np.nan


def screening_factor(sol, x, A, beta, rho_in, rho_bg, eps_edge=0.015, dens_fn=None):
    """S = |y'| / |y'_lin|, the gradient relative to its unscreened value.

    S -> 1 is unscreened and S -> 0 fully screened.
    """
    _check(sol)
    x = np.asarray(x, dtype=float)
    yp = sol.sol(x)[1]
    yp_lin = mass_charge(x, rho_in, rho_bg, eps_edge, dens_fn) / (A * x**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(yp) / np.abs(yp_lin)


def stability(sol, x, A, beta):
    """Perturbative stability of a solved profile.

    Fluctuations propagate on G^{mu nu} = -A g^{mu nu} + 2 beta (grad grad y -
    box y g^{mu nu}), giving

        G^tt      = A + 2 beta box y          (no ghost for > 0)
        c_r^2     = (A x + 4 beta y') / (x G^tt)
        c_theta^2 = (A + 2 beta (y'' + y'/x)) / G^tt

    Returns a dict of these on `x`; both sound speeds tend to 1 in the linear
    regime.
    """
    _check(sol)
    x = np.asarray(x, dtype=float)
    yp = sol.sol(x)[1]
    box = np.gradient(x**2 * yp, x) / x**2
    ypp = np.gradient(yp, x)
    Gtt = A + 2.0 * beta * box
    with np.errstate(divide="ignore", invalid="ignore"):
        cr2 = (A * x + 4.0 * beta * yp) / (x * Gtt)
        cth2 = (A + 2.0 * beta * (ypp + yp / x)) / Gtt
    return dict(x=x, Gtt=Gtt, c_r2=cr2, c_theta2=cth2)
