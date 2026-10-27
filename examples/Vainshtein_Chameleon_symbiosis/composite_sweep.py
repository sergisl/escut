"""Screening factor over the (radius, central density) plane, with the three
characteristic radii per density.

Each density is warm-started from the previous one: a cold solve stalls in the
mid-density range where the core is deep but the Vainshtein sphere has not yet
taken over. Writes `composite_data.npz`; the figure notebook reads it.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mixed_screening as ms

A = 1 + 6 * 0.4**2        # conformal coupling alpha = 0.4, so at = 1.96
A_PURE = 1.0              # shift-symmetric pure Galileon (no conformal coupling)
BETA = 50.0
BG = 3e-4
EPS_EDGE = 0.05
X_MAX = 6000.0

DC_SLICE = 7000.0         # density of the 1D slice in the lower panel

# Base grid, refined around the slice so the reference line lands on solved rows.
dcs = np.unique(np.concatenate([np.logspace(0.5, 6, 80),
                                np.logspace(np.log10(3500), np.log10(7500), 26)]))
xg = np.logspace(-1, np.log10(4000), 360)
XI = np.unique(np.concatenate([np.logspace(-2, np.log10(X_MAX), 4500),
                               np.linspace(0.5, 4.0, 1400)]))

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'composite_data.npz')


def run(tol=1e-3, max_nodes=2_000_000):
    n = len(dcs)
    M = np.full((n, len(xg)), np.nan)
    xvs = np.full(n, np.nan)
    xvp = np.full(n, np.nan)
    xs = np.full(n, np.nan)
    ok = np.zeros(n, dtype=bool)

    seed = None
    t0 = time.time()
    for i, dc in enumerate(dcs):
        s = None
        for attempt in ('warm', 'cold'):           # fall back if the seed does not take
            if attempt == 'warm' and seed is None:
                continue
            try:
                s = ms.solve(A=A, beta=BETA, rho_in=dc, rho_bg=BG, eps_edge=EPS_EDGE,
                             x_init=XI, tol=tol, max_nodes=max_nodes,
                             seed=seed if attempt == 'warm' else None)
                if s.success:
                    break
            except Exception:
                s = None
        if s is not None and s.success:
            ok[i] = True
            M[i] = ms.screening_factor(s, xg, A, BETA, dc, BG, EPS_EDGE)
            xvs[i] = ms.vainshtein_radius(xg, ms.solved_charge(s, xg), A, BETA)
            xs[i] = ms.core_radius(s, dc)
            yq = s.sol(s.x)
            seed = (s.x, yq[0], ms.solved_charge(s, s.x))
        xvp[i] = ms.vainshtein_radius_analytic(A_PURE, BETA, dc, BG)
        if (i + 1) % 10 == 0 or i == n - 1:
            print('density %d/%d  converged %d  (%.0fs)'
                  % (i + 1, n, ok[:i + 1].sum(), time.time() - t0), flush=True)

    np.savez(OUT, M=M, xvs=xvs, xvp=xvp, xs=xs, dcs=dcs, xg=xg, okmask=ok,
             A=A, A_pure=A_PURE, beta=BETA, BG=BG, eps_edge=EPS_EDGE)
    return M, xvs, xvp, xs, ok


if __name__ == '__main__':
    *_, ok = run()
    print('DONE  (%d/%d converged)' % (ok.sum(), len(ok)))
