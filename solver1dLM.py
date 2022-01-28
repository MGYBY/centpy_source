from ._helpers import *
from .equations import Equation1d
import sys

def source_mom_1(h, q, up, alpha, beta):
    ho = 3.0*(h-q/up)
    return (1.0/beta*(h-alpha*(1.0)-2.0*up/ho))


def source_mom_2(h, q, up, alpha, beta):
    ho = 3.0*(h-q/up)
    return (1.0/beta*(1.0-alpha*(1.0)/(h-ho)))

# problem-specific params
# included in the package in an ugly way
alpha_coeff = 0.30
beta_coeff = 27.0

class Solver1dLM:
    def __init__(self, equation):
        for key in equation.__dict__.keys():
            setattr(self, key, equation.__dict__[key])

        # Time step is determined in the solver
        self.dt = 0.0

        # Set the solver for the time step
        if self.scheme == "sd2":
            self.step = self.sd2
        elif self.scheme == "sd3":
            self.step = self.sd3
        elif self.scheme == "fd2":
            self.step = self.fd2
        else:
            sys.exit(
                "Scheme "
                + self.scheme
                + " is not recognized! Choices are: fd2, sd2, sd3."
            )

        # Equation dependent functions
        self.flux_x = equation.flux_x
        self.boundary_conditions = equation.boundary_conditions
        self.spectral_radius_x = equation.spectral_radius_x

        # # The unknown
        self.u = equation.initial_data()
        self.u_n = np.zeros((self.Nt + 1,) + self.u.shape)  # output array
        self.u_n[0] = self.u

        # CHANGE: source term initialization as zero
        # self.source = np.zeros_like(self.u)
        self.source = equation.source

    def H_flux(self, u_E, u_W, flux, spectral_radius):
        a = np.maximum(spectral_radius(u_E), spectral_radius(u_W))
        f_E = flux(u_E)
        f_W = flux(u_W)
        if u_W.shape == a.shape:
            return 0.5 * (f_W + f_E) - 0.5 * a * (u_W - u_E)  # scalar
        else:
            return 0.5 * (f_W + f_E) - 0.5 * np.multiply(
                a[:, None], (u_W - u_E)
            )  # for systems

    def c_flux(self, u_E, u_W):
        Hx_fluxp = self.H_flux(u_E[j0], u_W[jp], self.flux_x, self.spectral_radius_x)
        Hx_fluxm = self.H_flux(u_E[jm], u_W[j0], self.flux_x, self.spectral_radius_x)
        return -self.dt / self.dx * (Hx_fluxp - Hx_fluxm)

    #################
    # FD2
    #################

    def fd2(self, u, t):
        u_prime = np.ones(u.shape)
        un_half = np.ones(u.shape)
        self.boundary_conditions(u, t)
        f = self.flux_x(u)
        u_prime[1:-1] = limiter(u)
        # Predictor
        un_half[1:-1] = u[1:-1] - 0.5 * self.dt / self.dx * limiter(f)
        f_half = self.flux_x(un_half)
        # Corrector
        if self.odd:
            u[1:-2] = (
                0.5 * (u[2:-1] + u[1:-2])
                + 0.125 * (u_prime[1:-2] - u_prime[2:-1])
                - self.dt / self.dx * (f_half[2:-1] - f_half[1:-2])
            )
        else:
            u[2:-1] = (
                0.5 * (u[2:-1] + u[1:-2])
                + 0.125 * (u_prime[1:-2] - u_prime[2:-1])
                - self.dt / self.dx * (f_half[2:-1] - f_half[1:-2])
            )
        # Boundary conditions
        self.boundary_conditions(u, t)
        # Switch
        self.odd = not self.odd
        return u

    #################
    # SD2
    #################

    def reconstruction_sd2(self, u, t):
        # Reconstruction
        u_E = np.ones(u.shape)
        u_W = np.ones(u.shape)
        s = limiter(u[1:-1])
        u_E[j0] = u[j0] + 0.5 * s
        u_W[j0] = u[j0] - 0.5 * s
        self.boundary_conditions(u_E, t)
        self.boundary_conditions(u_W, t)
        return u_E, u_W

    def sd2(self, u, t):
        self.boundary_conditions(u, t)
        u_E, u_W = self.reconstruction_sd2(u, t)
        C0 = self.c_flux(u_E, u_W)
        u[j0] += C0
        self.boundary_conditions(u, t)
        u_E, u_W = self.reconstruction_sd2(u, t)
        C1 = self.c_flux(u_E, u_W)
        u[j0] += 0.5 * (C1 - C0)
        self.boundary_conditions(u, t)
        return u

    #################
    # SD3
    #################

    def reconstruction_sd3(self, u, ISl, ISc, ISr, t):
        cl = 0.25
        cc = 0.5
        cr = 0.25
        alpl = cl / ((eps + ISl) * (eps + ISl))
        alpc = cc / ((eps + ISc) * (eps + ISc))
        alpr = cr / ((eps + ISr) * (eps + ISr))
        alp_sum = alpl + alpc + alpr
        wl = alpl / alp_sum
        wc = alpc / alp_sum
        wr = alpr / alp_sum
        pl0, pl1, pr0, pr1, pc0, pc1, pc2 = p_coefs(u)
        u_E = np.ones(u.shape)
        u_W = np.ones(u.shape)
        u_E[j0] = (
            wl * (pl0 + 0.5 * pl1)
            + wc * (pc0 + 0.5 * pc1 + 0.25 * pc2)
            + wr * (pr0 + 0.5 * pr1)
        )
        u_W[j0] = (
            wl * (pl0 - 0.5 * pl1)
            + wc * (pc0 - 0.5 * pc1 + 0.25 * pc2)
            + wr * (pr0 - 0.5 * pr1)
        )
        # boundary
        self.boundary_conditions(u_E, t)
        self.boundary_conditions(u_W, t)
        return u_E, u_W

    # Implemented for one equation (for systems modify IS declarations)
    def sd3(self, u, t):
        self.boundary_conditions(u, t)
        u_norm = np.sqrt(self.dx) * np.linalg.norm(u[j0])
        pl0, pl1, pr0, pr1, pc0, pc1, pc2 = p_coefs(u)
        ISl = pl1 * pl1 / (u_norm + eps)
        ISc = 1.0 / (u_norm + eps) * ((13.0 / 3.0) * pc2 * pc2 + pc1 * pc1)
        ISr = pr1 * pr1 / (u_norm + eps)
        u_E, u_W = self.reconstruction_sd3(u, ISl, ISc, ISr, t)
        C0 = self.c_flux(u_E, u_W)
        u[2:-2] += +C0
        self.boundary_conditions(u, t)
        u_E, u_W = self.reconstruction_sd3(u, ISl, ISc, ISr, t)
        C1 = self.c_flux(u_E, u_W)
        u[j0] += +0.25 * (C1 - 3.0 * C0)
        self.boundary_conditions(u, t)
        u_E, u_W = self.reconstruction_sd3(u, ISl, ISc, ISr, t)
        C2 = self.c_flux(u_E, u_W)
        u[j0] += +1.0 / 12.0 * (8.0 * C2 - C1 - C0)
        self.boundary_conditions(u, t)
        return u

    ###################################################
    # source term integration using RK3TVD
    # for now, use a naive way to include the source term
    # integration sequence same as Toro's book
    ###################################################
    def source_int(self, u, t):
        arr_length = np.size(u[:,0])
        q_int_1 = 0.0
        up_int_1 = 0.0

        q_int_2 = 0.0
        up_int_2 = 0.0
        for i in range(2, arr_length-1):
            # first stage
            q_int_1 = u[i,1] + self.dt*(source_mom_1(u[i,0], u[i,1], u[i,2], alpha_coeff, beta_coeff))
            up_int_1 = u[i,2] + self.dt*(source_mom_2(u[i,0], u[i,1], u[i,2], alpha_coeff, beta_coeff))

            # second stage
            q_int_2 = (3.0/4.0)*u[i,1] + (1.0/4.0)*q_int_1 + (1.0/4.0)*self.dt*(source_mom_1(u[i,0], q_int_1, up_int_1,  alpha_coeff, beta_coeff))
            up_int_2 = (3.0/4.0)*u[i,2] + (1.0/4.0)*up_int_1 + (1.0/4.0)*self.dt*(source_mom_2(u[i,0], q_int_1, up_int_1,  alpha_coeff, beta_coeff))

            # third stage
            u[i,1] = (1.0/3.0)*u[i,1] + (2.0/3.0)*q_int_2 + (2.0/3.0)*self.dt*(source_mom_1(u[i,0], q_int_2, up_int_2, alpha_coeff, beta_coeff))
            u[i,2] = (1.0/3.0)*u[i,2] + (2.0/3.0)*up_int_2 + (2.0/3.0)*self.dt*(source_mom_2(u[i,0], q_int_2, up_int_2, alpha_coeff, beta_coeff))

        self.boundary_conditions(u, t)
        return u


    # Main solver routine
    def solve(self):
        i = 0
        t = 0.0
        t_out = 0.0
        while t < self.t_final:
            dt = self.set_dt()
            self.dt = min(dt, self.dt_out - t_out)
            t += self.dt
            t_out += self.dt
            self.u = self.step(self.u, t)
            self.u = self.source_int(self.u, t)
            max_h = np.max(self.u[:,0])
            max_h_ind = np.argmax(self.u[:,0])
            f1 = open("maxH.txt", "a+")
            f1.write("%18.8e %18.8e %18.8e \n" % (t, (self.dx*(max_h_ind-0.50)), max_h))
            # Store if t_out=dt_out
            if t_out == self.dt_out:
                i += 1
                self.u_n[i, :] = self.u
                t_out = 0

    def set_dt(self):
        r_max = np.max(self.spectral_radius_x(self.u))
        dt = self.dx * self.cfl / r_max
        return dt
