# This is where the parameters are stored, along with the grid.
# One can solve different equations using the same setup.
# That's why this class definition is separate from Equation.

from dataclasses import dataclass

# 1d
@dataclass
class Pars1d:
    # Grid parameters
    x_init: float = 0.0
    x_final: float = 1.0
    t_final: float = 1.0
    dt_out: float = 0.05
    J: int = 10
    cfl: float = 0.9
    scheme: str = "sd3"  # can be fd2, sd2, or sd3
    source: str = "rk3"
    iter_method: str = "hybr"
    n_coeff: float = 1.0
    strang_splitting: bool = False
    grav: float = 9.81
    mun: float = 0.0
    rho_mud: float = 0.0
    slope: float = 0.0

# 2d
@dataclass
class Pars2d(Pars1d):
    # Grid parameters in the 2nd dimension
    y_init: float = 0.0
    y_final: float = 1.0
    K: int = 10
