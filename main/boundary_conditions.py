"""Robin boundary conditions for the scalar BVP: Qx(0)=0 (regularity) and Qx+k(Q-Q_inf)=0 (far-field)."""

import numpy as np
from typing import Callable, Tuple


def outer_bc_robin_coeffs(A: float, B: float, C: float, xmax: float, Q_inf: float) -> float:
    """Robin BC coefficient k for Qx + k(Q - Q_inf) = 0 at x = xmax.

    k = m_eff + 1/xmax, where m_eff = sqrt(max(0, -(B + 2C·Q_inf)/A)).
    """
    # Estimate effective mass at outer boundary
    mu_out = -(B + 2.0*C*Q_inf)/A if A != 0 else 0.0
    m_out  = np.sqrt(mu_out) if mu_out > 0 else 0.0
    # Robin coefficient combines Yukawa decay + 1/r falloff
    return m_out + 1.0/xmax


def bc_robin(ya: np.ndarray, yb: np.ndarray, Q_inf: float, k: float) -> np.ndarray:
    """BVP boundary conditions: Qx(0)=0 (inner regularity), Qx+k(Q-Q_inf)=0 (outer Robin)."""
    # Inner: regularity
    # Outer: Robin
    return np.array([ya[1], yb[1] + k * (yb[0] - Q_inf)])


def bc_robin_jac(ya: np.ndarray, yb: np.ndarray, Q_inf: float, k: float) -> Tuple[np.ndarray, np.ndarray]:
    """Jacobian of bc_robin w.r.t. ya and yb. Returns (dya, dyb), each shape (2, 2)."""
    # ∂bc_inner/∂(Q_a, Qx_a) and ∂bc_inner/∂(Q_b, Qx_b)
    dya = np.array([[0, 1],      # inner BC: d/d_a(Q_x(0)) = 1
                    [0, 0]])     # outer BC: d/d_a(...) = 0
    # ∂bc_outer/∂(Q_b, Qx_b)
    dyb = np.array([[0, 0],      # inner BC: d/d_b(...) = 0
                    [k, 1]])     # outer BC: d/d_b(Q_x + k(Q - Q_inf)) = [k, 1]
    return dya, dyb
