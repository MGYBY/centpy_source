# Some examples in jupyter-notebook
1. Clear-water roll waves
2. Bingham mud model by Liu & Mei. Estimation of max. wave speed may weakly influence results. Don't know how to fix it, since the eigenstructure has no explicit solution (use [ConservationLawsDiffEq.jl](https://github.com/Paulms/ConservationLawsDiffEq.jl)?  use [pypde](https://pypde.readthedocs.io/en/latest/index.html)? or use implicit treatment of the stiff source term (implicit Euler or Crank-Nicolson)? ). One method to fix: use the numerical eignvalue of the Jacobian.
