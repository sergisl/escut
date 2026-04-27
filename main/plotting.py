"""Figures for the scalar BVP: plot_double_panel (flux + slope) and plot_profiles_and_effective_mass_combined (Q + m_eff²)."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from typing import Optional, Union
from scipy.signal import savgol_filter
from physics_utils import _cross_at_one, mu_for_tanh_step, compute_n_slope

# --- Standardized plotting sizes (use these across the module) ---
DEFAULT_LABEL_FS = 18
DEFAULT_TICK_FS = 14
DEFAULT_LEGEND_FS = 14
DEFAULT_CBAR_FS = 14

def set_default_plot_style(label_fs=DEFAULT_LABEL_FS, tick_fs=DEFAULT_TICK_FS,
                           legend_fs=DEFAULT_LEGEND_FS, cbar_fs=DEFAULT_CBAR_FS):
    """Apply a consistent plotting style across the module.

    This sets common rcParams so functions that don't pass explicit sizes
    inherit the same appearance.
    """
    plt.rcParams.setdefault('axes.labelsize', label_fs)
    plt.rcParams.setdefault('xtick.labelsize', tick_fs)
    plt.rcParams.setdefault('ytick.labelsize', tick_fs)
    plt.rcParams.setdefault('legend.fontsize', legend_fs)
    plt.rcParams.setdefault('figure.titlesize', max(label_fs, 14))
    plt.rcParams.setdefault('axes.titlesize', max(label_fs, 12))
    # Some Matplotlib versions don't expose a top-level 'colorbar.labelsize' rcParam
    if 'colorbar.labelsize' in plt.rcParams:
        plt.rcParams.setdefault('colorbar.labelsize', cbar_fs)

# apply defaults on import
set_default_plot_style()


# =============================================
# Private colormap helper
# =============================================
def _cmap_setup(sols_full, cmap_name, vmin, vmax):
    """Extract delta_c from solutions and build a (cmap, vmin, vmax, color_fn) tuple."""
    cmap = cm.get_cmap(cmap_name)
    n = len(sols_full)
    delta_cs = np.full(n, np.nan)
    for i, sol in enumerate(sols_full):
        meta = getattr(sol, "meta", {}) or {}
        dc = meta.get("delta_c", None)
        if dc is not None:
            delta_cs[i] = float(dc)
    finite = np.isfinite(delta_cs) & (delta_cs > 0)
    if vmin is None and finite.any():
        vmin = float(np.log10(delta_cs[finite].min()))
    if vmax is None and finite.any():
        vmax = float(np.log10(delta_cs[finite].max()))
    def color_for_dc(dc):
        if not (np.isfinite(dc) and dc > 0 and vmin is not None and vmax is not None and vmax > vmin):
            return "0.6"
        return cmap(np.clip((np.log10(dc) - vmin) / (vmax - vmin), 0.0, 1.0))
    return cmap, vmin, vmax, color_for_dc, delta_cs


# =============================================
# Paper figure 1: scalar flux + Vainshtein slope
# =============================================
def plot_double_panel(
    x, sols_full, *,
    linrefs=None,
    labels=None,
    cmap_name="viridis",
    lw_num=2.0,
    lw_lin=1.2,
    lin_ls=":",
    smooth_win_deriv: Union[int, float] = 61,
    smooth_poly_deriv: int = 3,
    smooth_win_vain: Union[int, float] = 0.03,
    smooth_poly_vain: int = 3,
    S=1.0,
    delta_inf=0.0,
    eps_edge=0.02,
    figsize=(8, 8),
    vmin=None,
    vmax=None,
    vradius: str = "full",
    vline_alpha: float = 0.5,
    vline_lw: float = 1.0,
    vline_ls: str = "striped",
    show_horizontal_guides: bool = True,
    ylim_deriv=None,
    save=None,
    label_fontsize: int = DEFAULT_LABEL_FS,
    tick_fontsize: int = DEFAULT_TICK_FS,
    cbar_fontsize: int = DEFAULT_CBAR_FS,
):
    """
    Two-panel paper figure: scalar flux x²|Q_x| (top, log-log) + local
    power-law slope n(x) (bottom, semi-log x).  Shared x-axis and δ_c colorbar.

    Parameters
    ----------
    x : array_like          Evaluation grid (log-spaced recommended).
    sols_full : list        Full (screened) BVP solutions.
    linrefs : list, optional  Matching linear solutions — plotted dotted.
    labels : list, optional   Legend labels (full curves only).
    smooth_win_deriv        Savitzky-Golay window for flux smoothing.
    smooth_win_vain         Window for n(x) smoothing (float fraction or int).
    S, delta_inf, eps_edge  Physical parameters for Vainshtein radius estimate.
    vradius : 'full'|'mu'|'none'  Which Vainshtein radius to mark.
    ylim_deriv              Optional (ymin, ymax) for the top flux panel.
    """
    x = np.asarray(x)
    nsol = len(sols_full)
    if labels is None:
        labels = [None] * nsol
    if linrefs is not None and len(linrefs) != nsol:
        raise ValueError("linrefs must be None or the same length as sols_full")

    cmap, vmin, vmax, color_for_dc, delta_cs = _cmap_setup(sols_full, cmap_name, vmin, vmax)
    eps = 1e-300

    # --- Figure ---
    fig = plt.figure(figsize=figsize)
    fig.patch.set_alpha(0)
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1], hspace=0.1)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    # --- Smoothing window normalization ---
    def _norm_win(win, nx):
        if isinstance(win, float) and 0 < win < 1:
            win = int(max(5, (nx * win) // 2 * 2 + 1))
        if isinstance(win, int) and win % 2 == 0:
            win += 1
        return win

    win_d = _norm_win(smooth_win_deriv, len(x))
    win_n = _norm_win(smooth_win_vain, len(x))

    xV_map = {}

    for i, sol in enumerate(sols_full):
        col = color_for_dc(delta_cs[i])
        meta = getattr(sol, "meta", {}) or {}
        A = meta.get("A", None)
        F = meta.get("F", None)
        dc = meta.get("delta_c", None)

        try:
            Qx = sol.sol(x)[1]
        except Exception:
            Qx = np.asarray(sol(x))[1]

        # ---- TOP: derivative ----
        y = x**2 * np.abs(Qx)
        if isinstance(win_d, int) and win_d >= 5:
            y = np.exp(savgol_filter(np.log(y + eps), win_d, smooth_poly_deriv))
        ax_top.loglog(x, y, color=col, lw=lw_num, label=labels[i])

        if linrefs is not None:
            try:
                Qx_lin = linrefs[i].sol(x)[1]
            except Exception:
                Qx_lin = np.asarray(linrefs[i](x))[1]
            y_lin = x**2 * np.abs(Qx_lin)
            if isinstance(win_d, int) and win_d >= 5:
                y_lin = np.exp(savgol_filter(np.log(y_lin + eps), win_d, smooth_poly_deriv))
            ax_top.loglog(x, y_lin, color=col, lw=lw_lin, linestyle=lin_ls, alpha=0.95, label=None)

        # ---- BOTTOM: n-slope ----
        L = np.log(x**2 * np.abs(Qx) + eps)
        Ls = savgol_filter(L, win_n, smooth_poly_vain) if isinstance(win_n, int) and win_n >= 5 else L
        n_vec = np.gradient(Ls, np.log(x))
        ax_bot.plot(x, n_vec, color=col, lw=lw_num, linestyle="-")

        if linrefs is not None:
            try:
                Qx_lin_n = linrefs[i].sol(x)[1]
            except Exception:
                Qx_lin_n = np.asarray(linrefs[i](x))[1]
            Llin = np.log(x**2 * np.abs(Qx_lin_n) + eps)
            Llin_s = savgol_filter(Llin, win_n, smooth_poly_vain) if isinstance(win_n, int) and win_n >= 5 else Llin
            ax_bot.plot(x, np.gradient(Llin_s, np.log(x)), color=col, lw=max(0.8, lw_lin), linestyle=":", alpha=0.9)

        # ---- Vainshtein radius ----
        xV_full = None
        if (A is not None) and (F is not None):
            R_arr = (4.0 * float(F) * np.abs(Qx)) / (float(A) * x + eps)
            xV_full = _cross_at_one(x, R_arr)
        xV_mu = None
        if (dc is not None) and (dc > 0):
            try:
                mu_tot = mu_for_tanh_step(x, dc, delta_inf, eps_edge, S)
                xV_mu = float(((8.0 * (F or 1.0) * mu_tot) / ((A or 1.0)**2))**(1.0/3.0))
            except Exception:
                pass
        xV_map[i] = {"xV_full": xV_full, "xV_mu": xV_mu, "delta_c": dc}

        xv_draw = xV_full if vradius == "full" else (xV_mu if vradius == "mu" else None)
        if xv_draw is not None:
            ls_style = (0, (4, 2)) if vline_ls == "striped" else vline_ls
            for ax_ in (ax_top, ax_bot):
                ax_.axvline(xv_draw, color=col, ls=ls_style, lw=float(vline_lw),
                            alpha=float(vline_alpha), label="_noref")

    # ---- Decorations ----
    for ax_ in (ax_top, ax_bot):
        ax_.axvline(1.0, color="k", ls="--", lw=1.0, alpha=0.85)
        ax_.tick_params(labelsize=tick_fontsize)
        ax_.grid(False)
        x_arr = np.atleast_1d(x)
        ax_.set_xlim(float(x_arr[0]), float(x_arr[-1]))

    ax_top.set_ylabel(r"Scalar flux $x^2|Q_x|$", fontsize=label_fontsize)
    if ylim_deriv is not None:
        ax_top.set_ylim(ylim_deriv)

    if show_horizontal_guides:
        for val, ls in [(3.0, ":"), (1.5, "--"), (0.0, "-.")]:
            ax_bot.axhline(val, color="gray", ls=ls, lw=1, alpha=0.6)
    ax_bot.set_xscale("log")
    ax_bot.set_xlabel(r"$x\equiv r/R$", fontsize=label_fontsize)
    ax_bot.set_ylabel(r"Slope $n$", fontsize=label_fontsize)

    ax_top.label_outer()
    ax_bot.label_outer()

    fig.tight_layout()
    finite = np.isfinite(delta_cs) & (delta_cs > 0)
    if finite.any() and vmin is not None and vmax is not None and vmax > vmin:
        sm = ScalarMappable(cmap=cmap)
        sm.set_clim(float(vmin), float(vmax))
        fig.subplots_adjust(right=0.86)
        cbar = fig.colorbar(sm, ax=[ax_top, ax_bot], pad=0.02, fraction=0.05, aspect=30)
        cbar.set_label(r"$\log_{10}(\delta_c)$", fontsize=cbar_fontsize)
        cbar.ax.tick_params(labelsize=tick_fontsize)

    if save:
        fig.savefig(save, dpi=180, bbox_inches="tight", transparent=True)
    return fig, (ax_top, ax_bot), xV_map


# =============================================
# Paper figure 2: Q profiles + effective scalar mass
# =============================================
def plot_profiles_and_effective_mass_combined(
    x, sols_full, *,
    B: Optional[float] = None,
    C: Optional[float] = None,
    gamma: float = 1.0,
    sign: float = -1.0,
    linrefs=None,
    anrefs=None,
    labels=None,
    cmap_name="viridis",
    lw_num=2.0,
    lw_lin=1.2,
    lin_ls=":",
    lw_an=1.5,
    an_ls="--",
    figsize=(8, 5.5),
    vmin=None,
    vmax=None,
    yscale_meff: str = "log",
    linthresh_meff: float = 1e-3,
    ylim_profiles=None,
    ylim_meff=None,
    twin_axes: bool = False,
    save=None,
    label_fontsize: int = DEFAULT_LABEL_FS,
    tick_fontsize: int = DEFAULT_TICK_FS,
    cbar_fontsize: int = DEFAULT_CBAR_FS,
):
    """Paper figure: |Q(x)| (left axis) and m_eff²/H² = sign·(B+C·Q)/γ (right axis).

    twin_axes=False  Two stacked panels (Q top, m_eff² bottom), shared x-axis.
    twin_axes=True   Single panel with secondary y-axis; B and C must be uniform.
    anrefs           Pre-computed analytical Q arrays (e.g. from analytic_Q_thin_shell),
                     plotted as dashed curves alongside the numerical solutions.

    Parameters
    ----------
    x : array_like
        Evaluation grid (linear or log-spaced).
    sols_full : list
        Full (screened) BVP solutions.
    B, C : float, optional
        Mass coefficients for m_eff² = sign·(B + C·Q)/γ.  Read from sol.meta if omitted.
    gamma : float
        Kinetic prefactor Γ (default 1).
    sign : float
        Sign convention (default −1, giving m² = −(B+CQ)/Γ).
    anrefs : list of ndarray, optional
        Pre-computed |Q| arrays, one per solution — plotted as dashed analytical curves.
    linrefs : list, optional
        Matching linearised BVP solutions — plotted dotted.
    twin_axes : bool
        If True, single-panel twin-axis layout; if False (default), two stacked panels.
    yscale_meff : {'log', 'symlog', 'linear'}
        Scale for the m_eff² panel (two-panel mode only).
    ylim_profiles, ylim_meff : tuple, optional
        Manual axis limits for the Q and m_eff² panels respectively.
    save : str, optional
        File path to save the figure.
    """
    if gamma == 0:
        raise ValueError("gamma must be non-zero.")
    x = np.asarray(x)
    nsol = len(sols_full)
    if labels is None:
        labels = [None] * nsol
    if linrefs is not None and len(linrefs) != nsol:
        raise ValueError("linrefs must be None or the same length as sols_full")
    if anrefs is not None and len(anrefs) != nsol:
        raise ValueError("anrefs must be None or the same length as sols_full")

    cmap, vmin, vmax, color_for_dc, delta_cs = _cmap_setup(sols_full, cmap_name, vmin, vmax)

    # ------------------------------------------------------------------ #
    # twin_axes=True: single panel, Q left / m_eff² right                 #
    # ------------------------------------------------------------------ #
    if twin_axes:
        # Resolve uniform B, C for the secondary axis transform
        meta0 = getattr(sols_full[0], "meta", {}) or {}
        B_ref = float(B if B is not None else meta0.get("B", np.nan))
        C_ref = float(C if C is not None else meta0.get("C", np.nan))
        if not (np.isfinite(B_ref) and np.isfinite(C_ref)):
            raise ValueError("B and C must be provided (or in sol.meta) for twin_axes mode.")

        def _q_to_meff2(q):
            return float(sign) * (B_ref + C_ref * np.asarray(q, dtype=float)) / float(gamma)

        def _meff2_to_q(m):
            return (np.asarray(m, dtype=float) * float(gamma) / float(sign) - B_ref) / C_ref

        fig, ax = plt.subplots(figsize=figsize)

        for i, sol in enumerate(sols_full):
            col = color_for_dc(delta_cs[i])
            try:
                Q = np.abs(sol.sol(x)[0])
            except Exception:
                Q = np.abs(np.asarray(sol(x))[0])
            ax.plot(x, Q, color=col, lw=lw_num, label=labels[i])
            if anrefs is not None:
                ax.plot(x, np.abs(np.asarray(anrefs[i])), color=col, lw=lw_an, linestyle=an_ls, alpha=0.9)
            if linrefs is not None:
                try:
                    Q_lin = np.abs(linrefs[i].sol(x)[0])
                except Exception:
                    Q_lin = np.abs(np.asarray(linrefs[i](x))[0])
                ax.plot(x, Q_lin, color=col, lw=lw_lin, linestyle=lin_ls, alpha=0.95)

        ax.axvline(1.0, color="k", ls="--", lw=1.0, alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlabel(r"$x\equiv r/R$", fontsize=label_fontsize)
        ax.set_ylabel(r"$|Q|$", fontsize=label_fontsize)
        ax.tick_params(labelsize=tick_fontsize)
        ax.grid(False)
        x_arr = np.atleast_1d(x)
        ax.set_xlim(float(x_arr[0]), float(x_arr[-1]))
        if ylim_profiles is not None:
            ax.set_ylim(ylim_profiles)

        if C_ref != 0:
            secax = ax.secondary_yaxis("right", functions=(_q_to_meff2, _meff2_to_q))
            secax.set_ylabel(r"$m_{\mathrm{eff}}^2 / H^2$", fontsize=label_fontsize)
            secax.tick_params(labelsize=tick_fontsize)

        legend_lines = [Line2D([0], [0], color="0.2", lw=lw_num, linestyle="-", label="Numerical")]
        if anrefs is not None:
            legend_lines.append(Line2D([0], [0], color="0.2", lw=lw_an, linestyle=an_ls, label="Analytical"))
        if linrefs is not None:
            legend_lines.append(Line2D([0], [0], color="0.2", lw=lw_lin, linestyle=lin_ls, label="Linear"))
        if len(legend_lines) > 1:
            ax.legend(handles=legend_lines, fontsize=DEFAULT_LEGEND_FS, loc="upper right")

        finite = np.isfinite(delta_cs) & (delta_cs > 0)
        if finite.any() and vmin is not None and vmax is not None and vmax > vmin:
            sm = ScalarMappable(cmap=cmap)
            sm.set_clim(float(vmin), float(vmax))
            # Place colorbar just to the right of the secondary y-axis
            fig.subplots_adjust(right=0.60)
            cbar = fig.colorbar(sm, ax=ax, pad=0.20, fraction=0.04, aspect=28)
            cbar.set_label(r"$\log_{10}(\delta_c)$", fontsize=cbar_fontsize)
            cbar.ax.tick_params(labelsize=tick_fontsize)
        else:
            fig.tight_layout()

        if save:
            fig.savefig(save, dpi=180, bbox_inches="tight", transparent=True)
        return fig, ax

    # ------------------------------------------------------------------ #
    # Default: two stacked panels                                          #
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.1)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    for i, sol in enumerate(sols_full):
        col = color_for_dc(delta_cs[i])
        meta = getattr(sol, "meta", {}) or {}
        B_i = float(B if B is not None else meta.get("B", np.nan))
        C_i = float(C if C is not None else meta.get("C", np.nan))
        if not (np.isfinite(B_i) and np.isfinite(C_i)):
            raise ValueError(f"B and C must be provided or available in sol.meta (sol {i}).")

        try:
            Q = sol.sol(x)[0]
        except Exception:
            Q = np.asarray(sol(x))[0]

        # ---- TOP: Q profiles ----
        ax_top.plot(x, Q, color=col, lw=lw_num, label=labels[i])

        # ---- BOTTOM: m_eff² ----
        ax_bot.plot(x, float(sign) * (B_i + C_i * Q) / float(gamma),
                    color=col, lw=lw_num, label=None)

        if linrefs is not None:
            meta_lin = getattr(linrefs[i], "meta", {}) or {}
            B_lin = float(B if B is not None else meta_lin.get("B", B_i))
            C_lin = float(C if C is not None else meta_lin.get("C", C_i))
            try:
                Q_lin = linrefs[i].sol(x)[0]
            except Exception:
                Q_lin = np.asarray(linrefs[i](x))[0]
            ax_top.plot(x, Q_lin, color=col, lw=lw_lin, linestyle=lin_ls, alpha=0.95, label=None)
            ax_bot.plot(x, float(sign) * (B_lin + C_lin * Q_lin) / float(gamma),
                        color=col, lw=lw_lin, linestyle=lin_ls, alpha=0.95, label=None)

    # ---- Decorations ----
    for ax_ in (ax_top, ax_bot):
        ax_.axvline(1.0, color="k", ls="--", lw=1.0, alpha=0.85)
        ax_.tick_params(labelsize=tick_fontsize)
        ax_.grid(False)
        x_arr = np.atleast_1d(x)
        ax_.set_xlim(float(x_arr[0]), float(x_arr[-1]))

    ax_top.set_ylabel(r"$Q$", fontsize=label_fontsize)
    if ylim_profiles is not None:
        ax_top.set_ylim(ylim_profiles)

    ax_bot.set_xlabel(r"$x\equiv r/R$", fontsize=label_fontsize)
    ax_bot.set_ylabel(r"$m_{\mathrm{eff}}^2$", fontsize=label_fontsize)
    if yscale_meff == "log":
        ax_bot.set_yscale("log")
    elif yscale_meff == "symlog":
        ax_bot.set_yscale("symlog", linthresh=linthresh_meff)
    if ylim_meff is not None:
        ax_bot.set_ylim(ylim_meff)

    ax_top.label_outer()
    ax_bot.label_outer()

    # Style legend: line meaning only
    legend_lines = [
        Line2D([0], [0], color="0.2", lw=lw_num, linestyle="-", label="Nonlinear"),
        Line2D([0], [0], color="0.2", lw=lw_lin, linestyle=lin_ls, label="Linear"),
    ]
    ax_top.legend(handles=legend_lines, fontsize=DEFAULT_LEGEND_FS, loc="best")

    fig.tight_layout()
    finite = np.isfinite(delta_cs) & (delta_cs > 0)
    if finite.any() and vmin is not None and vmax is not None and vmax > vmin:
        sm = ScalarMappable(cmap=cmap)
        sm.set_clim(float(vmin), float(vmax))
        fig.subplots_adjust(right=0.86)
        cbar = fig.colorbar(sm, ax=[ax_top, ax_bot], pad=0.02, fraction=0.05, aspect=30)
        cbar.set_label(r"$\log_{10}(\delta_c)$", fontsize=cbar_fontsize)
        cbar.ax.tick_params(labelsize=tick_fontsize)

    if save:
        fig.savefig(save, dpi=180, bbox_inches="tight", transparent=True)
    return fig, (ax_top, ax_bot)

