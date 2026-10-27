"""Void resolution vs depth: sweep (rho_in, beta) at fixed eps, recording the
minimum discriminant over x.

Plotted in the (rho_in, beta/(eps*at)^2) plane, where the pure-Vainshtein
boundary beta/(eps*at)^2 = 3/[8(rho_bg - rho_in)] is an eps-independent
analytic curve. Writes `void_depth.npz`; the figure notebook reads it.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mixed_screening as ms

alpha = 1.0 / np.sqrt(6.0)          # f(R) value, so at = 2
at = 1 + 6 * alpha**2
eps = 0.1
A = eps * at
BG = 1.0

# 140 x 76 rather than a square grid: the figure's top panel has a ~2:1 aspect,
# so a square grid would render as wide rectangles.
rho_ins = np.linspace(0.05, 0.99, 140)
betas = np.geomspace(0.0018, 0.32, 76)
XI = np.unique(np.concatenate([np.logspace(-2, 2, 3000), np.linspace(0.5, 20, 800)]))
XG = np.logspace(-1.2, np.log10(60), 3000)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'void_depth.npz')


def run():
    Dmin = np.full((len(betas), len(rho_ins)), np.nan)
    t0 = time.time()
    for j, IN in enumerate(rho_ins):
        seed = None          # march up in beta, carrying the previous solution:
        for i, F in enumerate(betas):          # a cold ramp per pixel is ~100x slower
            sol = None
            for attempt in ('warm', 'cold'):
                if attempt == 'warm' and seed is None:
                    continue
                try:
                    r = ms.solve(A=A, beta=F, rho_in=IN, rho_bg=BG, eps_edge=0.05,
                                 x_init=XI, tol=1e-4, max_nodes=5_000_000,
                                 seed=seed if attempt == 'warm' else None)
                    if r.success:
                        sol = r; break
                except Exception:
                    pass
            if sol is not None:
                Dmin[i, j] = float(np.nanmin(
                    ms.discriminant_ratio(XG, ms.solved_charge(sol, XG), A, F)))
                seed = (sol.x, sol.sol(sol.x)[0], ms.solved_charge(sol, sol.x))
        np.savez(OUT, rho_ins=rho_ins, betas=betas, Dmin=Dmin,
                 eps=eps, at=at, BG=BG)
        print('rho_in %d/%d (%.0fs)' % (j + 1, len(rho_ins), time.time() - t0), flush=True)
    return Dmin


if __name__ == '__main__':
    run()
    print('DONE')
