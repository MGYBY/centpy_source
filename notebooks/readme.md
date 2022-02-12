# Some examples in jupyter-notebook
1. Clear-water roll waves
2. Bingham mud model by Liu & Mei. Estimation of max. wave speed may weakly influence results. Don't know how to fix it, since the eigenstructure has no explicit solution (use [ConservationLawsDiffEq.jl](https://github.com/Paulms/ConservationLawsDiffEq.jl)?  use [pypde](https://pypde.readthedocs.io/en/latest/index.html)? or use implicit treatment of the stiff source term (implicit Euler or Crank-Nicolson)? ). One method to fix: use the numerical eignvalue of the Jacobian.
3. - [ ] Power-law fluid.

TODOs:

- [x] Use pypde.
- [x] Strang splitting.
- [x] Implicit Runge-Kutta for the source term: Pareschi & Russo's 2nd-order method, but the coefficients could be adjusted to Qin & Zhang.
- [x] Implicit Euler scheme (backward) for the source term.
- [x] Crank-Nicolson scheme for the source term.

