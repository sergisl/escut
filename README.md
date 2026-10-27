# escut

<p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/escut-dark.png?v=1">
    <source media="(prefers-color-scheme: light)" srcset="assets/escut-light.png?v=1">
    <img alt="escut logo" src="assets/escut-light.png" width="170">
  </picture>
</p>

**E**quation for **SC**reening with **U**nified **T**reatment &nbsp;·&nbsp; *(Escut means shield in Catalan)*

A Python library for numerically solving the master equation for screening in luminal Horndeski gravity, including the Vainshtein, Chameleon and Phaedrus mechanisms.

The code solves a nonlinear, dimensionless, spherically symmetric
boundary-value problem (BVP) for a scalar $Q(x)$, with $x = r/R$ the radial
coordinate in units of the source radius $R$, sourced by a density profile
$\tilde\rho(x)$ (a smoothed top hat by default). Schematically,

$$
\mathcal{K}[Q] \\; = \\; S\\,x^2\\,\tilde\rho(x) \\; + \\; x^2\\,\mathcal{V}(Q),
$$

where $\mathcal{K}$ collects the kinetic operators and $\mathcal{V}$ is the
contribution of the scalar's potential.

| Coefficient | Physics |
|---|---|
| `A` | Linear kinetic term |
| `D`, `E` | Phaedrus screening |
| `F` | Vainshtein screening |
| `S` | Source |

---

## Two scalar set-ups

What $Q$ represents, and hence the form of $\mathcal{V}$, is a modelling choice.
The two set-ups below are *different equations*, not two ways of writing one.

**Perturbative.** $Q$ is the perturbation of the scalar about a fixed
background, and the potential is Taylor-expanded about it and truncated,

$$
\mathcal{V}(Q) = Q\\,(B + CQ + GQ^2).
$$

The expansion coefficients then play the role of masses and self-couplings
(`B`, `C` set the Chameleon sector). This is the set-up of
[2605.04154](https://arxiv.org/abs/2605.04154), and is what the `B`, `C`, `G`
arguments mean.

**Full field.** $Q$ is the scalar itself, not a perturbation, and
$\mathcal{V}$ is supplied directly as an arbitrary callable `source_fn(x, Q)`
with no expansion. This is necessary whenever the field moves far from its
background — a deeply screened core or void — or whenever the potential has no
useful polynomial truncation. In this set-up `B`, `C`, `G` do not apply, and the
background value is an asymptotic field value rather than a linearisation point.

> **Sign convention.** `source_fn` must be written for
> $\mathcal{I}' = -x^2[S\tilde\rho + \mathcal{V}(Q)]$ as defined below. A
> potential written for the opposite convention, $\mathcal{I}' = +x^2 \mathcal{V}$,
> also requires the sign of `F` to be flipped, since the Vainshtein term enters
> $\mathcal{I}$ as $-2FxQ_x^2$. Getting this wrong does not fail loudly: it
> silently solves a different equation, usually one with $\Delta < 0$ over part
> of the domain, which shows up as the solver exhausting `max_nodes`.

---

## Two formulations

Given either equation, it can be integrated in two ways, selected with the
`method` argument of `solve_scalar_bvp`. Both return the same interface —
`.sol(x)` gives `[Q, Q_x]` — so downstream code need not know which was used.

```python
sol = solve_scalar_bvp(A=1.0, B=3.0, C=0.2, F=0.1, delta_c=5.0, method="gradient")
sol = solve_scalar_bvp(A=1.0, B=3.0, C=0.2, F=0.1, delta_c=5.0, method="charge")
```

**`method="gradient"`** (default) integrates the state $(Q, Q_x)$, solving
directly for $Q_{xx}$. This requires dividing by $(A + DQ)x^2 - 4F x Q_x$,
which can vanish near the Vainshtein transition.

**`method="charge"`** integrates the state $(Q, \mathcal{I})$ instead, where
$\mathcal{I}$ is the enclosed scalar charge obtained by integrating the kinetic
sector once:

$$
\mathcal{I} = x^2 (A + DQ)Q_x - 2F x Q_x^2, \qquad
\mathcal{I}' = -x^2\left[S\tilde\rho(x) + \mathcal{V}(Q)\right].
$$

The $\mathcal{I}$ equation has no denominator, and $Q_x$ is recovered from
$\mathcal{I}$ algebraically — a quadratic in $Q_x$ when $F \neq 0$, whose
discriminant $\Delta = (A+DQ)^2x^4 - 8Fx\mathcal{I}$ is exposed as
`charge_discriminant`. A negative $\Delta$ means no real $Q_x$ exists at that
state. This formulation is exact when `E == D`, and is generally the more robust
of the two at non-zero `F`; where both converge they agree to $\sim10^{-10}$.

`solve_scalar_bvp_charge_log` additionally works in $u = \ln Q$, for strictly
positive profiles spanning many decades.

---

## Installation

Clone the repo and install in editable mode (a virtual environment is recommended):

```bash
git clone https://github.com/Hi-COLACode/escut.git
cd escut
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

Dependencies: `numpy`, `scipy`, `matplotlib` (all installed automatically).

---

## Examples

Worked examples live in [`examples/`](examples/), one folder per topic.

[`single-mechanisms/`](examples/single-mechanisms/) takes each mechanism on its own —
[Chameleon](examples/single-mechanisms/chameleon-screening.ipynb),
[Vainshtein](examples/single-mechanisms/vainshtein-screening.ipynb) and
[Phaedrus](examples/single-mechanisms/phaedrus-screening.ipynb) — plus
[convergence tests](examples/single-mechanisms/convergence-tests.ipynb) of the solver.

[`Vainshtein_Chameleon_symbiosis/`](examples/Vainshtein_Chameleon_symbiosis/) is a
worked study using the charge formulation — two mechanisms acting at once, with a
notebook reproducing each figure of the accompanying paper.

---

## Citation

If you use this code, please cite the associated paper: [2605.04154](https://arxiv.org/abs/2605.04154).

---

## Contact

For questions or comments, please contact sergi.sirera@port.ac.uk.

> Parts of this codebase were developed with the assistance of large language models. All physics formulations and scientific results were designed and verified by the authors.

---

![GitHub language count](https://img.shields.io/github/languages/count/sergisl/escut)
![GitHub top language](https://img.shields.io/github/languages/top/sergisl/escut?color=yellow)
![GitHub forks](https://img.shields.io/github/forks/sergisl/escut?style=social)
![GitHub Repo stars](https://img.shields.io/github/stars/sergisl/escut?style=social)