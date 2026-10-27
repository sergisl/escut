# The Symbiosis of Vainshtein and Chameleon Screening

Notebooks reproducing every figure of the paper, built on escut's charge
formulation. One notebook per figure, each runnable on its own.

| Notebook | Figure | Runtime |
|---|---|---|
| [`void-stack.ipynb`](void-stack.ipynb) | Void pathology and its resolution | 2.5 min |
| [`composite.ipynb`](composite.ipynb) | Screening factor across the (radius, density) plane | 1 min |
| [`phase-diagrams.ipynb`](phase-diagrams.ipynb) | Vainshtein reduction and Chameleon de-screening | 6 min + 4 min |
| [`stability.ipynb`](stability.ipynb) | Perturbative stability (no-ghost, sound speeds) | 1.5 min |
| [`robustness-void.ipynb`](robustness-void.ipynb) | Void resolution for a compensated profile | seconds |
| [`robustness-halo.ipynb`](robustness-halo.ipynb) | Halo results for NFW-like and cored profiles | 1 min |

Runtimes are the parameter sweeps, measured on a laptop. Each notebook caches
its sweep to an `.npz` beside it, so re-running to adjust a figure is immediate.
Every sweep marches along one axis with `seed=` continuation, which is what keeps
them to minutes: solving each pixel cold instead costs one to two orders of
magnitude more, because a cold solve ramps the Vainshtein term up from zero every
time. The de-screening map is the clearest case -- 100 x 100 pixels in ~4 minutes
warm-started, against hours cold, since each is a deeply screened core on a domain
thousands of source radii across.

Grids: 140 x 76 (void), 100 x 100 (reduction, de-screening and stability). The
void grid is deliberately not square: that panel has a ~2:1 aspect, so a square
grid would render as wide rectangles.

## Running them

Nothing to install beyond escut's own dependencies (`numpy`, `scipy`,
`matplotlib`). The notebooks add `../../main` to `sys.path`, so they work from a
clone without installing the package:

```bash
cd examples/Vainshtein_Chameleon_symbiosis
jupyter notebook void-stack.ipynb
```

## Layout

- **`mixed_screening.py`** — the model and its diagnostics: the density-tracking
  minimum, density profiles (top-hat, NFW-like, Burkert, compensated void), and
  the measured quantities (discriminant, Vainshtein radius, core de-screening,
  screening factor, perturbative stability).
- **`*_sweep.py`** — the parameter sweeps behind each figure, kept separate from
  the notebooks because they are the slow part.
- **`*.npz`** — cached sweep output. Safe to delete; the notebooks recompute.

## Two things worth knowing

**The sign convention.** escut's charge has the Vainshtein term entering as
`-2F x Qx^2`, the opposite of the convention used in the paper. `mixed_screening`
takes `beta` and maps it to escut's `F = -beta` internally, so work in `beta`
throughout and do not pass `F` directly. Getting this wrong does not fail
loudly — it silently solves a different equation, usually one with a negative
discriminant over part of the domain, which surfaces as the solver exhausting
`max_nodes`.

**Caches record whether they are complete.** The sweeps save after every column
so a long run survives interruption, which means a cache file can exist while
still half-written. Each one carries a `complete` flag and the notebooks refuse
to trust a cache without it, recomputing instead. Trusting a partial cache is
an easy way to read placeholder `NaN`s as if they were results.

## Notes on the numerics

- The field is solved for directly, not as a perturbation about a background,
  so the Chameleon potential enters exactly rather than Taylor-expanded. This
  matters wherever the field departs far from its background — a deeply
  screened core, or a void.
- Deep profiles span many decades in the field, so the solves run in `u = ln y`.
- Sweeps warm-start each solve from its neighbour. This is not only faster
  (the reduction sweep goes from hours to minutes) but is what lets the
  strongly-screened regimes converge at all; a few points additionally need a
  finer homotopy ramp, which the sweeps fall back to automatically.
- `NaN` in a sweep output means the quantity is undefined there — for instance
  no Vainshtein sphere outside the source. Failure to converge is tracked
  separately, so an unsolved point is never silently drawn as a physical result.

## Citation

If you use this code, please cite the paper and
[escut](https://github.com/sergisl/escut).
