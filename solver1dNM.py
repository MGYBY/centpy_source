from ._helpers import *
from .equations import Equation1d
import sys
import csv
from scipy import optimize
import numpy as np

def source_mom_2(h, q, n):
    return (h-np.power((q/(h**2)),n))

def source_mom_implicit(p, h0, q0, delta_t, n):
    # x stands for q^(n+1), y stands for up^(n+1)
    x = p
    return (x-(q0+delta_t*(h0-np.power((x/h0**2),n))))

def source_mom_imex_1(p, h0, q0, delta_t, n, gamma_coeff):
    # x stands for q^(n+1), y stands for up^(n+1)
    x = p
    return (x-(h0-np.power(((q0+delta_t*gamma_coeff*x)/h0**2),n)))

def source_mom_imex_2(p, h0, q0, k1, delta_t, n, gamma_coeff1, gamma_coeff2):
    # x stands for q^(n+1), y stands for up^(n+1)
    x = p
    return (x-(h0-np.power(((q0+delta_t*gamma_coeff1*k1+delta_t*gamma_coeff2*x)/h0**2),n)))

class Solver1dNM:
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
        elif self.scheme == "lx":
            self.step = self.lx
        else:
            sys.exit(
                "Scheme "
                + self.scheme
                + " is not recognized! Choices are: fd2, sd2, sd3."
            )

        # Set the solver for the source term integration scheme
        if self.source == "rk3":
            self.source_int = self.source_rk3
        elif self.source == "rk104":
            self.source_int = self.source_rk104
        elif self.source == "backward_euler":
            self.source_int = self.source_backward_euler
        elif self.source == "cn":
            self.source_int = self.source_crank_nicolson
        elif self.source == "pr":
            self.source_int = self.source_pr
        else:
            self.source_int = self.source_rk3

        # Set the method for the iterative solver
        # if self.iter_method == 

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
    # Lax-Friedrichs
    #################

    def lx(self, u, t):
        # u_prime = np.ones(u.shape)
        un_half = np.ones(u.shape)
        self.boundary_conditions(u, t)
        # f = self.flux_x(u)
        # u_prime[1:-1] = limiter(u)
        # Predictor
        un_half[1:-1] = u[1:-1] # - 0.5 * self.dt / self.dx * limiter(f)
        f_half = self.flux_x(un_half)
        # Corrector
        if self.odd:
            u[1:-2] = (
                0.5 * (u[2:-1] + u[1:-2])
                # + 0.125 * (u_prime[1:-2] - u_prime[2:-1])
                - self.dt / self.dx * (f_half[2:-1] - f_half[1:-2])
            )
        else:
            u[2:-1] = (
                0.5 * (u[2:-1] + u[1:-2])
                # + 0.125 * (u_prime[1:-2] - u_prime[2:-1])
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
    ###################################################
    def source_rk3(self, u, t):
        # arr_length = np.size(u[:,0])

        # q_int_1 = 0.0
        # in vectorized form
        q_int_1 = u[j0,1] + self.dt*(source_mom_2(u[j0,0], u[j0,1], self.n_coeff))
        q_int_2 = (3.0/4.0)*u[j0,1] + (1.0/4.0)*q_int_1 + (1.0/4.0)*self.dt*(source_mom_2(u[j0,0], q_int_1, self.n_coeff))
        u[j0,1] = (1.0/3.0)*u[j0,1] + (2.0/3.0)*q_int_2 + (2.0/3.0)*self.dt*(source_mom_2(u[j0,0], q_int_2, self.n_coeff))

        # for i in range(2, arr_length-1):
        #     # first stage
        #     q_int_1 = u[i,1] + self.dt*(source_mom_2(u[i,0], u[i,1], self.n_coeff))

        #     # second stage
        #     q_int_2 = (3.0/4.0)*u[i,1] + (1.0/4.0)*q_int_1 + (1.0/4.0)*self.dt*(source_mom_2(u[i,0], q_int_1, self.n_coeff))

        #     # third stage
        #     u[i,1] = (1.0/3.0)*u[i,1] + (2.0/3.0)*q_int_2 + (2.0/3.0)*self.dt*(source_mom_2(u[i,0], q_int_2, self.n_coeff))

        self.boundary_conditions(u, t)
        return u

    ###################################################
    # source term integration using RK104
    ###################################################
    def source_rk104(self, u, t):
        # arr_length = np.size(u[:,0])
        q1, q2 = u[j0,1], u[j0,1]
        for i in range(1, 5+1):
            q1 = q1 + self.dt/6.0*(source_mom_2(u[j0,0], q1, self.n_coeff))
        q2 = (1.0/25.0)*q2+(9.0/25.0)*q1
        q1 = 15.0*q2-5.0*q1

        for i in range(6, 9+1):
            q1 = q1 + self.dt/6.0*(source_mom_2(u[j0,0], q1, self.n_coeff))

        u[j0,1] = q2 + (3.0/5.0)*q1 + self.dt/10.0*source_mom_2(u[j0,0], q1, self.n_coeff)

        # for i in range(2, arr_length-1):
        #     q1, q2 = u[i,1], u[i,1]
        #     for i in range(1, 5+1):
        #         q1 = q1 + self.dt/6.0*(source_mom_2(u[i,0], q1, self.n_coeff))

        #     q2 = (1.0/25.0)*q2+(9.0/25.0)*q1
        #     q1 = 15.0*q2-5.0*q1
        #     for i in range(6, 9+1):
        #         q1 = q1 + self.dt/6.0*(source_mom_2(u[i,0], q1, self.n_coeff))
            
        #     u[i,1] = q2 + (3.0/5.0)*q1 + self.dt/10.0*source_mom_2(u[i,0], q1, self.n_coeff)

        self.boundary_conditions(u, t)
        return u

    ###################################################
    # source term integration using backward Euler
    ###################################################

    def source_backward_euler(self, u, t):
        arr_length = np.size(u[:,0])

        # use the iterative solver in scipy to solve the nonlinear equation set
        for i in range(2, arr_length-1):
            u[i,1] = optimize.fsolve(source_mom_implicit, (u[i,1]), args=(u[i,0], u[i,1], self.dt, self.n_coeff))
            #  sol = optimize.root(source_mom_implicit, [u[i,1]], args=(u[i,0], u[i,1], self.dt, self.n_coeff), tol=1e-8, method=self.iter_method)
            #  u[i,1], u[i,2] = sol.x

        self.boundary_conditions(u, t)
        return u

    ###################################################
    # source term integration using Crank-Nicolson
    ###################################################

    def source_crank_nicolson(self, u, t):
        arr_length = np.size(u[:,0])

        # use the iterative solver in scipy to solve the nonlinear equation set
        for i in range(2, arr_length-1):
            q_implicit = optimize.fsolve(source_mom_implicit, (u[i,1]), args=(u[i,0], u[i,1], self.dt, self.n_coeff))
            q_explicit = u[i,1]+self.dt*source_mom_2(u[i,0], u[i,1], self.n_coeff)
            u[i,1] = (q_explicit+q_implicit)/2.0

        self.boundary_conditions(u, t)
        return u

    ###################################################
    # source term integration using Pareschi and Russo's two-stage 2nd order Diagonally Implicit Runge–Kutta method
    ###################################################

    def source_pr(self, u, t):
        arr_length = np.size(u[:,0])
        gamma_coeff = 1-1.0/np.sqrt(2.0)

        for i in range(2, arr_length-1):
            k1 = optimize.fsolve(source_mom_imex_1, (source_mom_2(u[i,0], u[i,1], self.n_coeff)), args=(u[i,0], u[i,1], self.dt, self.n_coeff, gamma_coeff))
            k2= optimize.fsolve(source_mom_imex_2, (source_mom_2(u[i,0], u[i,1], self.n_coeff)), args=(u[i,0], u[i,1], k1, self.dt, self.n_coeff, (1-2.0*gamma_coeff), gamma_coeff))
            u[i,1] += (0.50*self.dt*(k1+k2))

        return u


    # Main solver routine
    def solve(self):
        i = 0
        it = 0
        t = 0.0
        t_out = 0.0
        while t < self.t_final:
            dt = self.set_dt()
            self.dt = min(dt, self.dt_out - t_out)
            t += self.dt
            t_out += self.dt
            if (self.strang_splitting==True):
                self.dt /= 2.0
                self.u = self.source_int(self.u, t)
                self.dt *= 2.0
                self.u = self.step(self.u, t)
                self.dt /= 2.0
                self.u = self.source_int(self.u, t)
            else:
                self.u = self.step(self.u, t)
                self.u = self.source_int(self.u, t)
            # max_h = np.max(self.u[:,0])
            # max_h_ind = np.argmax(self.u[:,0])
            # f1 = open("maxH.txt", "a+")
            # f1.write("%18.8e %18.8e %18.8e \n" % (t, (self.dx*(max_h_ind-0.50)), max_h))
            # Store if t_out=dt_out
            if t_out == self.dt_out:
                i += 1
                self.u_n[i, :] = self.u
                format_string_time = f"{t:.3f}"
                file_name = 'outXYZ_%s.txt' % format_string_time
                with open(file_name, 'w') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerows(zip( np.transpose(self.x), np.transpose(self.u[:, 0]), np.transpose(self.u[:, 1])/np.transpose(self.u[:, 0]), ((np.transpose(self.u[:, 0])**(self.n_coeff))*((np.transpose(self.u[:, 1])/np.transpose(self.u[:, 0]))**(2.0-self.n_coeff)))))
                t_out = 0

            it += 1
            # output max and min of FR every 20 steps
            # outputseries: t-x-hMax-reMax-hMin
            if it%60 == 1:
                if np.max(self.u[:,0]) > 100:
                    print("\rInstability not tolerable. End running ... ...", end="")
                    sys.exit()
                max_h = np.max(self.u[:,0])
                max_h_ind = np.argmax(self.u[:,0])
                min_fr_h = 0.0
                count = max_h_ind
                max_u = self.u[count,1]/self.u[count, 0]
                max_re = (max_u**(2.0-self.n_coeff))*(max_h**(self.n_coeff))
                while (count>0):
                    # filter out numerical oscillations
                    if (self.u[count-1,0]>self.u[count,0] and self.u[count-2,0]>self.u[count-1,0] and 
                    self.u[count+1,0]<self.u[count,0] and self.u[count+2,0]<self.u[count+1,0] and
                    np.abs(self.u[count-1,0]-self.u[count,0])>2.0*np.abs(self.u[count,0]-self.u[count+1,0])):
                        min_fr_h = self.u[count,0]
                        break
                    count -= 1
                f1 = open("maxminH.txt", "a+")
                f1.write("%18.8e %18.8e %18.8e %18.8e %18.8e \n" % (t, (self.dx*(max_h_ind-0.50)), max_h, max_re, min_fr_h))

    def set_dt(self):
        r_max = np.max(self.spectral_radius_x(self.u))
        dt = self.dx * self.cfl / r_max
        return dt
