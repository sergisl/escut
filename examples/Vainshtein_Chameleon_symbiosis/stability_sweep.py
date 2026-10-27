"""Perturbative stability over the (Chameleon range, Vainshtein strength) plane.

Fluctuations propagate on an effective metric built from the background profile,

    G^tt      = A + 2 beta box(y)                (no ghost while positive)
    c_theta^2 = (A + 2 beta (y'' + y'/x)) / G^tt (angular sound speed)

Each pixel records the minimum of each over radius. Writes `stability_data.npz`;
the figure notebook reads it.
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
AT = 1 + 6 * ALPHA**2          # = 2 for the canonical f(R) coupling
EPS_EDGE = 0.05

N = 100
epss = np.geomspace(0.01, 10.0, N)
betas = np.geomspace(1e-4, 1e3, N)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stability_data.npz')


def _domain(eps, beta):
    """Outer radius and mesh, sized on the pure-Galileon Vainshtein scale."""
    xv = (8 * beta * RHO_IN / 3 / eps**2) ** (1 / 3.)
    xm = float(np.clip(9 * xv, 120, 4000))
    xi = np.unique(np.concatenate([np.logspace(-2, np.log10(xm), 3000),
                                   np.linspace(0.5, min(15, xm / 2), 1000)]))
    return xm, xi


def pixel(eps, beta, tol=3e-3, max_nodes=4_000_000, seed=None):
    """min_x G^tt and min_x c_theta^2 at one point of the plane, or None.

    A cold solve ramps beta up from zero for every pixel, which is where nearly
    all the cost sits. Passing `seed` (a previous solution) lets the solver go
    straight to the target, roughly two orders of magnitude faster.
    """
    A = eps * AT
    xm, xi = _domain(eps, beta)
    s = None
    if seed is not None:
        r = ms.solve(A=A, beta=beta, rho_in=RHO_IN, rho_bg=RHO_BG, eps_edge=EPS_EDGE,
                     x_init=xi, tol=tol, max_nodes=max_nodes, seed=seed)
        if r.success:
            s = r
    if s is None:                                  # cold fallback
        r = ms.solve(A=A, beta=beta, rho_in=RHO_IN, rho_bg=RHO_BG, eps_edge=EPS_EDGE,
                     x_init=xi, tol=tol, max_nodes=max_nodes)
        if r.success:
            s = r
    if s is None:
        return None

    xg = np.logspace(-1.5, np.log10(xm * 0.5), 2500)
    st = ms.stability(s, xg, A, beta)
    Gtt, cth2 = st['Gtt'], st['c_theta2']

    # Keep clear of both boundaries: the inner mesh edge and the outer BC.
    m = (xg > 0.08) & (xg < xm * 0.4) & np.isfinite(Gtt)
    if m.sum() < 10:
        return None
    minG = float(np.nanmin(Gtt[m]))

    # c_theta^2 only where the mode is not already ghost-unstable: it diverges
    # as G^tt -> 0, so one divergence would otherwise set the colour scale.
    healthy = m & (Gtt > 0) & np.isfinite(cth2)
    minC = (float(np.nanmin(np.clip(cth2[healthy], -10, 10)))
            if healthy.sum() >= 5 else np.nan)
    return minG, minC, (s.x, s.sol(s.x)[0], ms.solved_charge(s, s.x))


def run(tol=3e-3, max_nodes=4_000_000):
    Gmin = np.full((N, N), np.nan)
    Cmin = np.full((N, N), np.nan)
    t0 = time.time()
    for j, eps in enumerate(epss):
        seed = None                      # march up in beta, carrying the solution
        for i, beta in enumerate(betas):
            r = pixel(eps, beta, tol=tol, max_nodes=max_nodes, seed=seed)
            if r is not None:
                Gmin[i, j], Cmin[i, j], seed = r
        print('eps %d/%d  (%.0fs)' % (j + 1, N, time.time() - t0), flush=True)
    np.savez(OUT, epss=epss, betas=betas, Gmin=Gmin, Cmin=Cmin,
             rho_in=RHO_IN, rho_bg=RHO_BG, at=AT, eps_edge=EPS_EDGE)
    return Gmin, Cmin


if __name__ == '__main__':
    G, C = run()
    print('DONE  solved %d/%d;  ghost pixels %d,  angular-unstable %d'
          % (np.isfinite(G).sum(), G.size, (G < 0).sum(), (C < 0).sum()))
