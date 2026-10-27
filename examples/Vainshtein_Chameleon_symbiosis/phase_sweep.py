"""The two phase maps over the (Chameleon range, Vainshtein strength) plane.

`run_reduction()` — the Vainshtein radius relative to its pure-Galileon value,
    x_{V+Ch}/x_V. The comparison radius is the *shift-symmetric* Galileon (no
    conformal coupling, kinetic coefficient eps rather than eps*at), which is
    what makes the ratio saturate at the geometric ceiling at^(-2/3) rather
    than at 1. Writes `phase_reduction.npz`.

`run_descreening()` — the displacement of the central field from its interior
    minimum, D = (y_0 - y_in)/(y_bg - y_in). Writes `phase_descreening.npz`.

The two use different grids: the reduction map is smooth and worth sampling
finely, while each de-screening pixel is a deep-core solve and is sampled more
coarsely in beta.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mixed_screening as ms

RHO_IN = 300.0
RHO_BG = 3e-4
ALPHA = 1.0 / np.sqrt(6.0)
AT = 1 + 6 * ALPHA**2
EPS_EDGE = 0.05

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_RED = os.path.join(HERE, 'phase_reduction.npz')
OUT_DSC = os.path.join(HERE, 'phase_descreening.npz')


def _domain(eps, beta):
    """Outer radius and mesh: outside both the Vainshtein sphere and the
    background Compton wavelength, so the outer condition is imposed where the
    field has already reached its asymptote."""
    A = eps * AT
    xv = (8 * beta * RHO_IN / 3 / eps**2) ** (1 / 3.)
    lam = ms.compton_wavelength(A, RHO_BG)
    xm = float(np.clip(max(15 * xv, 10 * lam), 300, 20000))
    xi = np.unique(np.concatenate([np.logspace(-2, np.log10(xm), 4500),
                                   np.linspace(0.5, 30, 1500)]))
    return xm, xi


def run_reduction(n=100, tol=2e-3, max_nodes=8_000_000, resume=True):
    """Vainshtein-radius reduction over the plane.

    Each beta is warm-started from the one below it: the high-eps columns sit on
    a domain of many thousands of source radii, where cold solves are very slow.
    With `resume`, columns already present in the cache are kept, so an
    interrupted run continues rather than restarting.
    """
    epss = np.geomspace(0.01, 10.0, n)
    betas = np.geomspace(1e-4, 1e3, n)
    RAT = np.full((n, n), np.nan)
    XV = np.full((n, n), np.nan)
    # XV is NaN both where the mechanism is eliminated and where the solve
    # failed. Those mean different things -- one is a result, the other is a
    # gap -- so track convergence separately rather than letting the figure
    # paint an unsolved pixel as "eliminated".
    SOLVED = np.zeros((n, n), dtype=bool)
    failures = []
    LAM = np.array([[ms.compton_wavelength(e * AT, RHO_BG) for e in epss]] * n)

    j_start = 0
    if resume and os.path.exists(OUT_RED):
        old = np.load(OUT_RED)
        if old['RAT'].shape == RAT.shape and np.allclose(old['epss'], epss):
            RAT, XV = old['RAT'].copy(), old['XV'].copy()
            if 'SOLVED' in old.files:
                SOLVED = old['SOLVED'].copy()
            done = [j for j in range(n) if not np.isnan(RAT[:, j]).all()]
            j_start = (max(done) + 1) if done else 0
            print('resuming from column %d (%d already done)' % (j_start, j_start), flush=True)

    t0 = time.time()
    for j in range(j_start, n):
        eps = epss[j]
        A = eps * AT
        seed = None                     # warm start along beta within the column
        for i, beta in enumerate(betas):
            xm, xi = _domain(eps, beta)
            try:
                s = None
                # warm start is fastest; fall back to a cold solve, then to a
                # finer homotopy ladder, which rescues the few pixels where the
                # default ramp hits a singular Jacobian.
                for attempt in ('warm', 'cold', 'fine'):
                    if attempt == 'warm' and seed is None:
                        continue
                    s = ms.solve(A=A, beta=beta, rho_in=RHO_IN, rho_bg=RHO_BG,
                                 eps_edge=EPS_EDGE, x_init=xi, tol=tol,
                                 max_nodes=max_nodes,
                                 steps=40 if attempt == 'fine' else 6,
                                 seed=seed if attempt == 'warm' else None)
                    if s.success:
                        break
                if s is None or not s.success:
                    continue
                SOLVED[i, j] = True
                seed = (s.x, s.sol(s.x)[0], ms.solved_charge(s, s.x))
                xg = np.logspace(-1, np.log10(xm * 0.8), 4000)
                xvc = ms.vainshtein_radius(xg, ms.solved_charge(s, xg), A, beta)
                # NaN marks elimination: no Vainshtein sphere outside the source
                if np.isfinite(xvc) and xvc > 1.0:
                    XV[i, j] = xvc
                    RAT[i, j] = xvc / ms.vainshtein_radius_analytic(eps, beta, RHO_IN, RHO_BG)
            except Exception as exc:          # record, never swallow silently
                failures.append((eps, beta, f'{type(exc).__name__}: {exc}'))
        np.savez(OUT_RED, epss=epss, betas=betas, RAT=RAT, XV=XV, LAM=LAM,
                 SOLVED=SOLVED, alpha=ALPHA, at=AT, rho_in=RHO_IN, rho_bg=RHO_BG,
                 complete=(j == n - 1))
        print("eps %d/%d  (%.0fs)" % (j + 1, n, time.time() - t0), flush=True)
    unsolved = int((~SOLVED).sum())
    if unsolved:
        print('%d pixel(s) did not converge (reported as unsolved, not eliminated)'
              % unsolved, flush=True)
    if failures:
        print('%d pixel(s) raised; first: %s' % (len(failures), failures[0]), flush=True)
    return RAT, XV, LAM


def run_descreening(n_eps=100, n_beta=100, tol=2e-3, max_nodes=8_000_000):
    # Same grid as run_reduction, so both panels of the phase figure share one
    # frame and render at the same resolution.
    epss = np.geomspace(0.01, 10.0, n_eps)
    betas = np.geomspace(1e-4, 1e3, n_beta)
    D = np.full((len(betas), len(epss)), np.nan)
    Y0 = D.copy()
    failures = []

    t0 = time.time()
    for j, eps in enumerate(epss):
        A = eps * AT
        seed = None                 # warm start along beta within each eps column
        for i, beta in enumerate(betas):
            _, xi = _domain(eps, beta)
            try:
                s = None
                for attempt in ('warm', 'cold'):
                    if attempt == 'warm' and seed is None:
                        continue
                    s = ms.solve(A=A, beta=beta, rho_in=RHO_IN, rho_bg=RHO_BG,
                                 eps_edge=EPS_EDGE, x_init=xi, tol=tol,
                                 max_nodes=max_nodes,
                                 seed=seed if attempt == 'warm' else None)
                    if s.success:
                        break
                if s is not None and s.success:
                    Y0[i, j] = float(np.ravel(s.sol(np.atleast_1d(1e-2)))[0])
                    D[i, j] = ms.descreening(s, RHO_IN, RHO_BG)
                    seed = (s.x, s.sol(s.x)[0], ms.solved_charge(s, s.x))
            except Exception as exc:          # record, never swallow silently
                failures.append((eps, beta, f'{type(exc).__name__}: {exc}'))
        np.savez(OUT_DSC, epss=epss, betas=betas, D=D, Y0=Y0,
                 alpha=ALPHA, at=AT, rho_in=RHO_IN, rho_bg=RHO_BG,
                 complete=(j == len(epss) - 1))
        print('eps %d/%d  (%.0fs)' % (j + 1, len(epss), time.time() - t0), flush=True)
    if failures:
        print('%d pixel(s) raised; first: %s' % (len(failures), failures[0]), flush=True)
    return D, Y0


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if which in ('both', 'descreening'):
        D, _ = run_descreening()
        print('descreening done: %d/%d solved' % (np.isfinite(D).sum(), D.size))
    if which in ('both', 'reduction'):
        RAT, XV, _ = run_reduction()
        print('reduction done: %d/%d solved, %d eliminated'
              % (np.isfinite(RAT).sum(), RAT.size, np.isnan(XV).sum()))
