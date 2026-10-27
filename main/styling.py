"""Figure styling: journal widths, rcParams and a shared colour palette.

Draw each figure at its true print width (`COL` single-column, `WIDE`
two-column) so fonts render at their real point size rather than being shrunk
by \\includegraphics[width=\\linewidth].

    import styling
    styling.apply()
    fig, ax = plt.subplots(figsize=(styling.COL, 2.6))
    ax.pcolormesh(X, Y, Z, cmap=styling.sequential())
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
# Sizes
# ---------------------------------------------------------------------------

COL = 3.40      # APS single-column width (inches)
WIDE = 7.05     # APS two-column / figure* width (inches)
ANNOT = 8       # in-axes annotation font size (pt)


def apply():
    """Set rcParams for publication-sized figures."""
    plt.rcParams.update({
        'font.size':         9,
        'axes.labelsize':    10,
        'axes.titlesize':    10,
        'xtick.labelsize':   8.5,
        'ytick.labelsize':   8.5,
        'legend.fontsize':   8,
        'axes.linewidth':    0.8,
        'lines.linewidth':   1.6,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'figure.dpi':        130,
        'savefig.dpi':       220,
        'savefig.bbox':      'tight',
    })


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------

def _mk(name, stops):
    return LinearSegmentedColormap.from_list(name, stops, N=256)


# Sequential maps, all starting from black so the low end reads as "screened".
CMAPS = {
    'deepsea': _mk('deepsea', [(0, '#04060d'), (0.18, '#0c2340'), (0.40, '#15566a'),
                               (0.60, '#2b9080'), (0.76, '#8cc06a'), (0.86, '#bcd35f'),
                               (0.94, '#e6cf5f'), (1, '#fbf6d3')]),
    'ember':   _mk('ember',   [(0, '#060509'), (0.18, '#2a1143'), (0.38, '#6a1f63'),
                               (0.57, '#b13b55'), (0.74, '#e3673b'), (0.88, '#f6a94a'), (1, '#fdf2c4')]),
    'nebula':  _mk('nebula',  [(0, '#070512'), (0.20, '#241351'), (0.42, '#5a2178'),
                               (0.60, '#9c3a7f'), (0.76, '#d05f73'), (0.90, '#f0a07a'), (1, '#fde9cf')]),
    'aurora':  _mk('aurora',  [(0, '#03060a'), (0.20, '#0e2a4e'), (0.42, '#135e63'),
                               (0.62, '#2f9560'), (0.80, '#9cc24f'), (0.93, '#e8df7a'), (1, '#fafbe0')]),
    # Night-sky ramp: violet-black through indigo and cyan to a pale mint. Unlike
    # the maps above it never reaches yellow, so the bright end stays cold.
    'borealis': _mk('borealis', [(0, '#0a0714'), (0.18, '#221a4d'), (0.38, '#2f3f9e'),
                                 (0.58, '#2d8fc4'), (0.74, '#3fc9bf'), (0.88, '#8fe3b0'),
                                 (1, '#e9f8ec')]),
}

SEQUENTIAL = 'borealis'


def sequential(name=None):
    """The sequential colormap; defaults to `SEQUENTIAL`."""
    return CMAPS[name or SEQUENTIAL]


# Diverging map: warm (magma-like) below the midpoint, cool (deepsea) above, so
# negative reads as pathological and positive as healthy.
DIVERGING = _mk('diverging', [
    (0.00, '#6e1f55'), (0.14, '#a62f66'), (0.28, '#dc5a3a'), (0.40, '#f2a978'),
    (0.50, '#f3f0e8'),
    (0.60, '#a9d2ca'), (0.72, '#3a9f88'), (0.86, '#176074'), (1.00, '#0c2340'),
])

# Two encoding channels, kept separate:
#   linestyle  -> which theory (solid = Vainshtein + Chameleon, dashed = pure
#                 Vainshtein, dotted = pure Chameleon)
#   colour     -> which case (density profile, or position along a sweep)
# Giving theory its own hues as well meant six competing colours in one figure;
# neutral lines let the colourmap carry the data and the accents stay meaningful.

LINE = {
    'primary':   '#2b2b2b',   # curves over a light ground
    'secondary': '#6e6e6e',   # supporting curves, guides
    'onmap':     '#f2f2f2',   # curves drawn over a dark colourmap
}

LINESTYLE = {'vch': '-', 'v': '--', 'c': ':'}

# Categorical colours, sampled across the sequential map so they belong to the
# same family as the rest of the paper. Ordered dark -> light, which also reads
# correctly if printed greyscale.
# Brightened so the three stay separable at small size on a white panel; rust
# also reads as the warm (pathological) member of the set. Figures are free to
# assign these to cases in whatever order suits their own background.
CASES = ['#1d4e7c', '#3fc0a4', '#c2552f']   # blue, teal, rust

# Named uses of the same triplet, so one vocabulary covers every figure that
# shows three cases.
PROFILES = {'top-hat': CASES[0], 'NFW-like': CASES[1], 'cored': CASES[2]}

# For quantities ordered along a sweep (e.g. increasing Vainshtein strength).
# Warm, so they separate from the cool map they are drawn on, and ordered, so
# three steps read as a progression rather than three unrelated categories.
def ordered(n, cmap='magma', lo=0.35, hi=0.80):
    """`n` evenly spaced colours along a map, for an ordered sequence."""
    import numpy as _np
    m = plt.get_cmap(cmap)
    return [m(t) for t in _np.linspace(lo, hi, n)]
