# escut

**E**quation for **SC**reening with **U**nified **T**reatment

A lightweight Python library for numerically solving the master equation for screening in luminal Horndeski gravity, including the Vainshtein, Chameleon and Phaedrus mechanisms.

The code solves the dimensionless spherical boundary-value-problem (BVP) for a scalar field $Q(x)$, $x = r/R$:

$$
A\,Q_{xx} + \frac{2A}{x}Q_x + Q(B + CQ) - \frac{D}{x^2}\frac{d}{dx}[x^2 Q Q_x] - E\,Q_x^2 + \frac{4F}{x^2}\frac{d}{dx}[x\,Q_x^2] = -S\,\delta(x)
$$

with a smoothed top-hat density profile $\delta(x)$ and Robin boundary conditions matched to a Yukawa exterior.

| Mechanism | Driven by | Key coefficients |
|---|---|---|
| **Chameleon** | Nonlinear potential | `B`, `C` |
| **Vainshtein** | Derivative self-interaction | `F` |
| **Phaedrus** | Kinetic nonlinearity | `D`, `E` |

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
