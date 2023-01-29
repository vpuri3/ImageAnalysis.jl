#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 16:03:41 2022

@author: micbenn
"""

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

###################################
# Optimization options for the finite element form compiler
###################################
parameters["form_compiler"]["cpp_optimize"] = True
parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -ffast-math'
parameters["form_compiler"]["quadrature_degree"] = 3

set_log_level(LogLevel.ERROR)

###################################
# Create or read mesh
###################################

mesh = RectangleMesh(Point(-0.5,-0.5),Point(0.5,0.5), 20, 20)
cell_markers = MeshFunction("bool", mesh,2)
cell_markers.set_all(False)
for cell in cells(mesh):
    p = cell.midpoint()
    if abs(p[1]) < 0.25:
        cell_markers[cell] = True
mesh = refine(mesh, cell_markers)
cell_markers = MeshFunction("bool", mesh,2)
cell_markers.set_all(False)
for cell in cells(mesh):
    p = cell.midpoint()
    if abs(p[1]) < 0.15:
        cell_markers[cell] = True
mesh = refine(mesh, cell_markers)
cell_markers = MeshFunction("bool", mesh,2)
cell_markers.set_all(False)
for cell in cells(mesh):
    p = cell.midpoint()
    if abs(p[1]) < 0.1:
        cell_markers[cell] = True
mesh = refine(mesh, cell_markers)

###################################
# Model Setup
###################################

#########
# FEM setup
#########
V = VectorFunctionSpace(mesh, 'CG',1)
Vcg = FunctionSpace(mesh, 'CG', 1)
Vdg = FunctionSpace(mesh, 'DG', 1)

u = Function(V)         # displacement at t_i
u_old = Function(V)     # displacement from previous timestep
u_new = Function(V)     # displacement at t_i+1
d = Function(V)         # phase field variable at time t_i
d_old = Function(V)     # phase field variable at previous timestep
d_new = Function(V)     # phase field variable at time t_i+1

# initialize all fields to zero
z = Constant((0.0, 0.0))
u.interpolate(z)
u_old.interpolate(z)
u_new.interpolate(z)
d.interpolate(z)
d_old.interpolate(z)
d_new.interpolate(z)

## During iterative solving:
    # 1) use d to solve for u_new starting at u
    # 2) use u_new to solve for d_new
    # 3) reset u = u_new and d = d_new and iterate

u_  = TestFunction(V)   # test function for the displacement problem
d_  = TestFunction(V)   # test function for the phase field problem
du_ = TrialFunction(V)  # trial function for the displacement problem
dd_ = TrialFunction(V)  # trial function for the phase field problem

#########
# Initial Conditions
#########

class IC_d(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.w = args[0]    # crack width
    def eval(self, values, x):

        # crack
        if abs(x[1]) < self.w and x[0] <= 0:
            values[0] = 0
            values[1] = 1 
        else:
            values[0] = 0
            values[1] = 0

    def value_shape(self):
        return (2,)

w = 0.01
d_ic = IC_d(w, degree=2)
d.interpolate(d_ic)
d_old.interpolate(d_ic)
d_new.interpolate(d_ic)

#########
# Boundary Conditions
#########

def boundary_bottom(x, on_boundary):
    return on_boundary and near(x[1], -0.5, DOLFIN_EPS)

def top_boundary(x, on_boundary):
    return on_boundary and near(x[1], 0.5, DOLFIN_EPS)

load = Expression("t", t = 0.0, degree=1)

bc_u_B = DirichletBC(V.sub(1), Constant(0.0), boundary_bottom)
bc_u_T = DirichletBC(V.sub(1), load, boundary_bottom)

bcs_u = [bc_u_B, bc_u_T]
bcs_d = [] # leave empty

#########
# Parameters
#########

## Material parameters (normalized by 1e9 for scaling efficiency)
#mu_mat, lam_mat = 1.104,2.576   # intact material properties
mu, lmbda = 1, 0
mu_mat, lam_mat = 1.0, 0.0
#mu_d, lam_d = 1.104,2.576       # crack parameters for compression
mu_d, lam_d = 1.0, 0.0
#Gc,eps = 285 / 1e9 ,0.015
eps = 0.015
Gc = 1.721 / 100e3
eta_eps = 1e-3
Cr = 0.001
crack_formed_limit = 0.95 # magnitude of the phase field at which the material is considered cracked

## Solver parameters
dt = 0.01      # load step
T_max = 1       # maximum load applied to material

N = int(T_max/dt)
t = np.linspace(0,T_max,int(1/dt)) # load steps

err_max = 0.001

############################################################
def energy(alpha1, alpha0, beta):    
    Energy = mu/2 *(alpha0**2 + alpha1**2 +beta**2 -2)+ h(alpha0*alpha1)
    return Energy

def h(J):
    return (lmbda/2)*(J-1)**2 -mu*ln(J)

############################################################

# Strain energy function for the un-cracked material
def W_mat(u):
    I = Identity(2)           # identity tensor
    F = variable(grad(u) + I) # deformation gradient tensor
    C = F.T * F               # Cauchy deformation
    J = variable(det(F))      # determinant of deformation
    
    # strain energy of intact material
    W = (mu_mat/2)*(tr(C) - 2 - 2*ln(J)) + (lam_mat/2)*(J-1)**2 
    
    return W
#
# Strain energy function for the cracked material
def W_d(u,d):
    I = Identity(2)           # identity tensor
    F = variable(grad(u) + I) # deformation gradient tensor
    d = variable(d)
    J = det(F)                # determininant of deformation
    n1 = d[0] / sqrt(dot(d,d))# normalized components of normal vector
    n2 = d[1] / sqrt(dot(d,d))
    
    # getting components of triangularized matrix
    A11 = sqrt((F[0,0]*n2 - F[0,1]*n1)**2 + (F[1,0]*n2 - F[1,1]*n1)**2)
    A22 = J / A11

    A22_star = ( lam_d*A11 + sqrt( 4*mu_d**2 + 4*mu_d*lam_d*A11**2 + (lam_d*A11)**2 ) )/(2*(mu_d + lam_d*A11**2))

    W = conditional(lt(A22,A22_star),energy(A11,A22,0),energy(A11,A22_star,0))
    
    return W

# Energy from new surfaces
def E_surf(d):
    d_mag = dot(d,d)
    d_grd = grad(d)
    d_grd_mag_sq = inner(d_grd,d_grd)
    
    E = Gc*( d_mag / (2*eps) + (eps/2)*d_grd_mag_sq )
    
    return E

# energy density functional
def E_tot(u,d):
    # checking if mag of d < critical value
    d_mag = conditional(lt(dot(d,d),Cr),0,sqrt(dot(d,d)))
    
    E = ((1- d_mag)**2 + eta_eps)*W_mat(u) + (1-(1- d_mag)**2 )*\
        conditional(lt(dot(d,d), Cr),0.0, W_d(u,d)) +\
        E_surf(d)

    return E
############################################################

## Total potential energy functional
def Phi(u,d):
    P = E_tot(u,d)*dx
    return P


## Algorithm to prevent crack healing
def prevent_crack_heal(d_old,d_new):
    # if the material was previously cracked (|d|>0.95), make sure the material remains cracked
    dold_vals = d_old.vector().get_local() # nodal values for previous crack
    dnew_vals = d_new.vector().get_local() # nodal values for tenative new crack
    
    for i in range(0,len(dold_vals),2):
        # looping through all nodal values
        dold_mag = sqrt( (dold_vals[i])**2 + (dold_vals[i+1])**2 ) # previous crack mag
        if dold_mag > crack_formed_limit:
            dnew_vals[i] = dold_vals[i] / dold_mag
            dnew_vals[i+1] = dold_vals[i+1] / dold_mag
            
    # reassigning new crack function
    dnew_unheal = Function(V)
    dnew_unheal.vector()[:] = dnew_vals[:]
    return dnew_unheal
            

Phi_u = Phi(u,d_old)            # Potential energy functional for displacement problem
Phi_d = Phi(u_new,d)    # Potential energy functional for phase field problem

F_u = derivative(Phi_u,u,u_)        # derivative of energy wrt displacement
F_d = derivative(Phi_d,d,d_)    # derivative of energy wrt phase field
J_u = derivative(F_u,u,du_)         # Jacobian of energy wrt dispalcement
J_d = derivative(F_d,d,dd_)     # Jacobian of energy wrt displacement

###################################
# Nonlinear solver setup
###################################

# setting up the problem and solver
prob_u = NonlinearVariationalProblem(F_u,u,bcs_u,J_u)
prob_d = NonlinearVariationalProblem(F_d,d,bcs_d,J_d)
solve_u = NonlinearVariationalSolver(prob_u)
solve_d = NonlinearVariationalSolver(prob_d)

solve_u.parameters['nonlinear_solver'] = 'newton'
solve_d.parameters['nonlinear_solver'] = 'newton'

nlparams_u = solve_u.parameters['newton_solver']
nlparams_d = solve_d.parameters['newton_solver']

nlparams_u['maximum_iterations'] = 50
nlparams_d['maximum_iterations'] = 50

nlparams_d['report'] = True
nlparams_d['absolute_tolerance'] = 1e-6

nlparams_u['report'] = True
nlparams_u['absolute_tolerance'] = 1e-6

########## Solver Loop ##########
d_mag = dot(d,d)

CrackVector_file  = File("./Frac/crack_mag.pvd", "compressed")
Displacement_file = File("./Frac/displacement.pvd", "compressed")
TotalEnergy_file  = File("./Frac/total_energy.pvd", "compressed")

#CrackVector_file  << mesh
#Displacement_file << mesh
#TotalEnergy_file  << mesh

CrackVector_file  << (d, 0.0)
Displacement_file << (u, 0.0)
TotalEnergy_file  << (project(E_tot(u,d), Vdg), 0.0)

#p = plot(d_mag)
#plt.colorbar(p)
#plt.show()


# looping through all load steps
N = 100
for i in range(N):

    if MPI.rank(mesh.mpi_comm()) == 0:
        print('###################################')
        print(f'Iteration #{i}. Load: {float(t[i])}')
        print('###################################')

    # update load
    load.t = t[i]

    err = 1
    iter_n = 0  # number of iterations to converge
    while err > err_max:
        iter_n += 1
        
        # solve the displacement problem
        solve_u.solve() # solves displacement into u
        u_new.assign(u) # update the displacement solution for the crack problem
        
        # solve the crack problem
        solve_d.solve() # solves crack into d
        dnew = prevent_crack_heal(d_old, d) # ensures d doesn't heal, assign to d_new
        d_new.assign(d_new)
        error_u = errornorm(u_new,u_old) # change between iterations of displacement
        error_d = errornorm(d_new,d_old) # change between iterations of crack
        err = max(error_u,error_d) # getting maximum error (end iter if max err < err_max)

        u_old.assign(u_new)
        d_old.assign(d_new)
        
        if err < err_max:
            if MPI.rank(mesh.mpi_comm()) == 0:
                #if (i % 10 == 0):
                #    p = plot(d_mag)
                #    plt.colorbar(p)
                #    plt.show()

                print('###################################')
                print('Iteration Complete.')
                print('Number of iterations: ',iter_n)
                print('Current Error: ',err)
                print('###################################')

            if (i % 20 == 0):
                CrackVector_file  << (d, t[i])
                Displacement_file << (u, t[i])
                TotalEnergy_file  << (project(E_tot(u, d), Vdg), t[i])
            else:
                pass
        else:
            pass
#
