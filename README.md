# escut

<table style="border: none; border-collapse: collapse;" cellpadding="10">
<tr>
<td style="border: none;" width="160">
  <img src="assets/escut-light.png#gh-light-mode-only" alt="escut logo" width="150"/>
  <img src="assets/escut-dark.png#gh-dark-mode-only" alt="escut logo" width="150"/>
</td>
<td style="border: none;">

**E**quation for **SC**reening with **U**nified **T**reatment &nbsp;·&nbsp; *(Escut means shield in Catalan)*

A Python library for numerically solving the master equation for screening in luminal Horndeski gravity, including the Vainshtein, Chameleon and Phaedrus mechanisms.

</td>
</tr>
</table>

The code solves the nonlinear dimensionless spherical boundary-value-problem (BVP) for a scalar field perturbation $Q(x)$, $x = r/R$:

$$
\frac{1}{x^2}\frac{d}{dx}\left[x^2 \left(A + D Q\right) Q_x\right] + x^2 Q(B + CQ) - \frac{1}{x^2}\frac{d^2}{dx^2}\left[F x^2 Q_x^2\right] + E x^2 Q_x^2 + S x^2 \tilde{\rho}(x) = 0,
$$

with a smoothed top-hat density profile $\delta(x)$, where x is the dimensionless radial coordinate $x = r/R$, with $R$ being the radius of the source.

| Coefficient | Physics |
|---|---|
| `A` | Linear kinetic term |
| `B`, `C` | Chameleon screening |
| `D`, `E` | Phaedrus screening |
| `F` | Vainshtein screening |
| `S` | Source |

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

See the Jupyter notebooks for worked examples of each mechanism: [Chameleon](notebooks/chameleon-screening.ipynb), [Vainshtein](notebooks/vainshtein-screening.ipynb), and [Phaedrus](notebooks/phaedrus-screening.ipynb).

---

## Citation

If you use this code, please cite:

```bibtex
@article{Sirera2026master,
  author  = {Sirera, Sergi and Baker, Tessa and Hallam, James and Naidoo, Krishna},
  title   = {{A Master Equation for Screening in Luminal Horndeski Gravity}},
  journal = {arXiv},
  year    = {2026},
  note    = {arXiv:XXXX.XXXXX},
  url     = {https://arxiv.org/abs/XXXX.XXXXX},
}
```

---

## AI assistance disclosure

Parts of this codebase were developed with the assistance of GitHub Copilot. All physics formulations and scientific results were designed and verified by the authors.

---

## Contact

For questions or comments, please contact me at sergi.sirera@port.ac.uk.
