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
u_  = TestFunction(V)   # test function for the displacement problem
d_  = TestFunction(V)   # test function for the phase field problem
du_ = TrialFunction(V)  # trial function for the displacement problem
dd_ = TrialFunction(V)  # trial function for the phase field problem

dnew_temp = Function(V)

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

def bottom_boundary(x, on_boundary):
    return on_boundary and near(x[1], -0.5, DOLFIN_EPS)

def top_boundary(x, on_boundary):
    return on_boundary and near(x[1], 0.5, DOLFIN_EPS)

load = Expression("t", t = 0.0, degree=1)

bc_u_B = DirichletBC(V, Constant((0.0,0.0)), bottom_boundary) #boundary_bottom)
bc_u_T = DirichletBC(V.sub(1), load, top_boundary) #top_boundary)

bcs_u = [bc_u_B, bc_u_T]
bcs_d = [] # leave empty

Gc, eps, lmbda, mu, eta_eps =  1.721/ (100e3), 0.015, 0, 1, 1.e-3
Cr = 1.e-3
crack_formed_limit = 0.95 # magnitude of the phase field at which the material is considered cracked

def W_mat(u):
    I = Identity(2)           # identity tensor
    F = variable(grad(u) + I) # deformation gradient tensor
    C = F.T * F               # Cauchy deformation
    J = variable(det(F))      # determinant of deformation
    
    # strain energy of intact material
    W = (mu/2)*(tr(C) - 2 - 2*ln(J)) + (lmbda/2)*(J-1)**2 
    return W

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

    A22_star = ( lmbda*A11 + sqrt( 4*mu**2 + 4*mu*lmbda*A11**2 + (lmbda*A11)**2 ) )/(2*(mu + lmbda*A11**2))

    W_0 = mu/2 *(A11**2 + A22**2 - 2 - 2*ln(A11*A22)) + (lmbda/2)*(A11*A22 - 1)**2
    W_star = mu/2 *(A11**2 + A22_star**2 - 2 - 2*ln(A11*A22_star)) + (lmbda/2)*(A11*A22_star - 1)**2
    
    W = conditional(lt(A22,A22_star),W_0,W_star)
    return W

def E_surf(d):
    d_mag = dot(d,d)
    d_grad = grad(d)
    d_grad_mag = inner(grad(d),grad(d))

    E = Gc* ( d_mag/(2*eps) + (eps/2)*d_grad_mag )

    return E

def E_tot(u,d):
    E = ((1- conditional(lt(dot(d,d), Cr),0.0, sqrt(dot(d,d))))**2 + eta_eps)*W_mat(u) +\
        (1-(1- conditional(lt(dot(d,d), Cr),0.0, sqrt(dot(d,d))))**2 )*conditional(lt(dot(d,d), Cr),0.0, W_d(u,d)) +\
        E_surf(d)             
    return E

def E_elas(u,d):
    E = ((1- conditional(lt(dot(d,d), Cr),0.0, sqrt(dot(d,d))))**2 + eta_eps)*W_mat(u) +\
        (1-(1- conditional(lt(dot(d,d), Cr),0.0, sqrt(dot(d,d))))**2 )*conditional(lt(dot(d,d), Cr),0.0, W_d(u,d))              
    return E
        

def Phi(u,d):
    P = E_tot(u,d)*dx
    return P

Phi_u = Phi(u, d_old)          					    
Phi_d = Phi(u_new, d) 
F_u = derivative(Phi_u, u, u_)   
F_d = derivative(Phi_d, d, d_)
J_u = derivative(F_u, u, du_)  
J_d  = derivative(F_d, d, dd_)

p_u = NonlinearVariationalProblem(F_u, u, bcs_u, J_u)
p_d = NonlinearVariationalProblem(F_d, d, bcs_d ,J_d)
solver_u = NonlinearVariationalSolver(p_u)
solver_d = NonlinearVariationalSolver(p_d)

prm1 = solver_u.parameters
prm1['newton_solver']['maximum_iterations'] = 1000
prm2 = solver_d.parameters
prm2['newton_solver']['maximum_iterations'] = 1000

def prevent_crack_heal(d_old, d_new):  #conserves the direction of old crack
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
    dnew_temp.vector()[:] = dnew_vals[:]
    return dnew_temp


energy_total = Function(Vdg)

CrackVector_file = File ("./Frac/crack_vector.pvd")
Displacement_file = File ("./Frac/displacement.pvd")   
TotalEnergy_file = File ("./Frac/total_energy.pvd")

t = 0
u_r = 0.003
deltaT  = 0.1
tol = 1e-6

## Solver parameters
T_max = 1.8       # maximum load applied to material

t = 0
while t<= T_max: 
    if t>= 1.45:
        deltaT = 1.e-2
    elif t>=1.68:
        deltaT = 1e-3
    t += deltaT

    if t<1.8:
        load.t=t*u_r
    else:
        load.t=1.8*u_r

    iter = 0
    err = 1
    while err > tol:
        iter += 1
        solver_u.solve()
        u_new.assign(u) 

        solver_d.solve()
        dnew = prevent_crack_heal(d_old, d)
        d_new.assign(dnew)  

        err_u = errornorm(u_new,u_old,norm_type = 'l2',mesh = None)
        err_phi = errornorm(d_new,d_old,norm_type = 'l2',mesh = None)
        err = max(err_u,err_phi)
        
        u_old.assign(u_new)
        d_old.assign(d_new)

        if err <= tol:
            print ('Iterations:', iter, ', Total time', t)
            d_new.rename("d", "crack_vector")
            CrackVector_file << d_new
            Displacement_file << u_new

            energy_total = project( E_tot(u_new, d_new) ,Vdg)
            energy_total.rename("energy_total", "energy_total")
            TotalEnergy_file << energy_total