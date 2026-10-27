# One mechanism at a time

Each notebook takes a single screening mechanism on its own: the field profile,
the force it mediates, and how the screening turns on with source density.
They are the place to start — the symbiosis study next door assumes them.

| Notebook | What it covers |
|---|---|
| [`chameleon-screening.ipynb`](chameleon-screening.ipynb) | Density-dependent mass, thin-shell suppression |
| [`vainshtein-screening.ipynb`](vainshtein-screening.ipynb) | Derivative self-interaction, the Vainshtein radius |
| [`phaedrus-screening.ipynb`](phaedrus-screening.ipynb) | The Phaedrus mechanism |
| [`convergence-tests.ipynb`](convergence-tests.ipynb) | Mesh and tolerance convergence of the solver |

All four use the gradient formulation, `solve_scalar_bvp` over the state
`(Q, Q_x)`, as introduced in [arXiv:2605.04154](https://arxiv.org/abs/2605.04154). For two mechanisms acting at once — where that formulation runs into
a vanishing denominator and the charge formulation takes over — see
[`../Vainshtein_Chameleon_symbiosis/`](../Vainshtein_Chameleon_symbiosis/).

## Running them

Nothing to install beyond escut's dependencies (`numpy`, `scipy`, `matplotlib`).
Each notebook locates `main/` by searching upward for it, so they work from a
clone without installing the package, and from any working directory.
