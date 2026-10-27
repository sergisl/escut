"""Robustness of the results to the assumed density profile.

The main results use a smoothed top hat. These sweeps repeat them for realistic
shapes — an NFW-like cusp and a cored Burkert halo, and a compensated void with
an overdense ridge — to check that nothing depends on the idealised profile.

`run_void()` writes `robustness_void.npz`; `run_halo()` writes
`robustness_halo.npz`.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mixed_screening as ms

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_VOID = os.path.join(HERE, 'robustness_void.npz')
OUT_HALO = os.path.join(HERE, 'robustness_halo.npz')

# ---- void ----
V_IN, V_BG = 1.0 / 3.0, 1.0
V_A, V_BETA = 0.3, 0.08
V_EPS_EDGE = 0.05

# ---- halo ----
H_ALPHA = 0.5
H_AT = 1 + 6 * H_ALPHA**2          # = 2.5
H_EPS = 0.03
H_A = H_EPS * H_AT
H_BG = 3e-4
H_EPS_EDGE = 0.05
H_BETA_RED = 1.0                   # fixed beta for the reduction panel
H_IN_DSC = 300.0                   # fixed density for the de-screening panel


def void_profiles():
    return {'top-hat': ms.density_profile_fn(V_IN, V_BG, V_EPS_EDGE),
            'compensated': ms.compensated_void_profile(V_IN, V_BG)}


def halo_profiles(rho_in):
    return {'top-hat': ms.density_profile_fn(rho_in, H_BG, H_EPS_EDGE),
            'NFW-like': ms.nfw_profile(rho_in, H_BG),
            'cored': ms.burkert_profile(rho_in, H_BG)}


def run_void(x_max=40.0, tol=1e-6, max_nodes=2_000_000):
    """Discriminant for each void profile, pure-Galileon vs symbiotic."""
    x = np.logspace(-1.2, 1.3, 500)
    out = {'x': x}
    for name, dens in void_profiles().items():
        I_pure = ms.mass_charge(x, V_IN, V_BG, V_EPS_EDGE, dens_fn=dens)
        out[f'pure_{name}'] = ms.discriminant_ratio(x, I_pure, V_A, V_BETA)
        s = ms.solve(A=V_A, beta=V_BETA, rho_in=V_IN, rho_bg=V_BG,
                     eps_edge=V_EPS_EDGE, dens_fn=dens, x_min=1e-2, x_max=x_max,
                     tol=tol, max_nodes=max_nodes)
        out[f'mixed_{name}'] = (ms.discriminant_ratio(x, ms.solved_charge(s, x), V_A, V_BETA)
                                if s.success else np.full_like(x, np.nan))
        print(f'  void {name}: converged={s.success}', flush=True)
    np.savez(OUT_VOID, complete=True, A=V_A, beta=V_BETA,
             rho_in=V_IN, rho_bg=V_BG, eps_edge=V_EPS_EDGE, **out)
    return out


def _halo_domain(rho_in, beta, dens):
    """Outside both the pure-Galileon sphere and the background Compton range."""
    xr = np.logspace(-1, np.log10(2e4), 6000)
    I = ms.mass_charge(xr, rho_in, H_BG, H_EPS_EDGE, dens_fn=dens)
    xV = ms.vainshtein_radius(xr, I, H_EPS, beta)     # shift-symmetric comparison
    lam = ms.compton_wavelength(H_A, H_BG)
    xm = float(np.clip(max(15 * np.nan_to_num(xV, nan=5.0), 15 * lam), 5000, 20000))
    xi = np.unique(np.concatenate([np.logspace(-2, np.log10(xm), 4500),
                                   np.linspace(0.5, 30, 1500)]))
    return xV, xm, xi


def run_halo(tol=3e-3, max_nodes=6_000_000):
    """Vainshtein-radius reduction against density, and de-screening against beta."""
    rho_ins = np.geomspace(50., 1e4, 8)
    betas = np.geomspace(0.1, 5., 8)
    names = list(halo_profiles(H_IN_DSC))
    RED = {n: np.full(len(rho_ins), np.nan) for n in names}
    DSC = {n: np.full(len(betas), np.nan) for n in names}

    t0 = time.time()
    for k, rho_in in enumerate(rho_ins):           # reduction, fixed beta
        for name, dens in halo_profiles(rho_in).items():
            xV, xm, xi = _halo_domain(rho_in, H_BETA_RED, dens)
            s = ms.solve(A=H_A, beta=H_BETA_RED, rho_in=rho_in, rho_bg=H_BG,
                         eps_edge=H_EPS_EDGE, dens_fn=dens, x_init=xi,
                         tol=tol, max_nodes=max_nodes)
            if not s.success:
                continue
            xg = np.logspace(0, np.log10(xm * 0.9), 9000)
            xVC = ms.vainshtein_radius(xg, ms.solved_charge(s, xg), H_A, H_BETA_RED)
            if np.isfinite(xVC) and np.isfinite(xV):
                RED[name][k] = xVC / xV
        print('reduction %d/%d (%.0fs)' % (k + 1, len(rho_ins), time.time() - t0), flush=True)

    for k, beta in enumerate(betas):               # de-screening, fixed density
        for name, dens in halo_profiles(H_IN_DSC).items():
            _, _, xi = _halo_domain(H_IN_DSC, beta, dens)
            s = ms.solve(A=H_A, beta=beta, rho_in=H_IN_DSC, rho_bg=H_BG,
                         eps_edge=H_EPS_EDGE, dens_fn=dens, x_init=xi,
                         tol=tol, max_nodes=max_nodes)
            if s.success:
                DSC[name][k] = ms.descreening(s, H_IN_DSC, H_BG)
        print('de-screening %d/%d (%.0fs)' % (k + 1, len(betas), time.time() - t0), flush=True)

    np.savez(OUT_HALO, complete=True, rho_ins=rho_ins, betas=betas,
             at=H_AT, A=H_A, eps=H_EPS, rho_bg=H_BG,
             beta_red=H_BETA_RED, rho_in_dsc=H_IN_DSC,
             **{f'RED_{n}': RED[n] for n in names},
             **{f'DSC_{n}': DSC[n] for n in names})
    return RED, DSC


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if which in ('both', 'void'):
        run_void(); print('void done')
    if which in ('both', 'halo'):
        run_halo(); print('halo done')
