# compute the inf-sup constant for Stokes
# check to see if there is a compenent in Z at each step
# using iterated penalty

from firedrake import *
import math,sys


mu = Constant("1.0")
#mu = Constant("0.0")
#mu = Constant("0.6")
shft=mu
print("%.2e"%shft)
# create mesh and define velocity and pressure function spaces 
meshsize=int(sys.argv[1])
print("meshsize %s" % (meshsize))
kdeg = int(sys.argv[2])
print("kdeg %s" % (kdeg))
max_iters = int(sys.argv[3])
print("max_iters %s" % (max_iters))
kin = int(sys.argv[4])
print("kin %s" % (kin))
mswich=int(sys.argv[5])
print("mswich %s" % (mswich))
lam=1.0

if (mswich == 1):
    mesh = UnitSquareMesh(meshsize, meshsize)
    domainstr = "regmesh"
if (mswich == 2):
    mesh = UnitSquareMesh(meshsize, meshsize, diagonal="crossed")
    domainstr = "Malkus"
if (mswich == 3):
    segmnts=7
    domain =  Circle(dolfin.Point(0.0, 0.0),1.0,segmnts)
    mesh = generate_mesh(domain, meshsize)
    domainstr = "circle"
if (mswich == 4):
    mesh = UnitCubeMesh(meshsize, meshsize, meshsize)
    domainstr = "Cube"
if (mswich == 5):
    mesh = Mesh("cube-%i.msh" % meshsize)
    meshsize = 1./meshsize
    domainstr = "Cube-Unstructured"

V = VectorFunctionSpace(mesh, "Lagrange", kdeg)
# define boundary condition
if mesh.topological_dimension() == 2:
    boundary_exp = Constant((0., 0.))
    x, y = SpatialCoordinate(mesh)
    start = as_vector([sin(kin*y), cos(kin*x)])
else:
    boundary_exp = Constant((0., 0., 0.))
    x, y, z = SpatialCoordinate(mesh)
    start = as_vector([sin(kin*y*z), cos(kin*x*z), sin(kin*x*y)])
bc = DirichletBC(V, boundary_exp, "on_boundary")
# set the parameters
# define test and trial functions, and function that is updated
u = TrialFunction(V)
v = TestFunction(V)
#w = Function(V)
uold = Function(V)
uold = interpolate(start,V)
# set the variational problem
a = inner(grad(u), grad(v))*dx 
u = Function(V)
bo = div(uold)*div(v)*dx 
sp = {"ksp_type": "cg", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
a_solver = LinearSolver(assemble(a, bcs=bc), solver_parameters=sp)
a_solver.solve(u, assemble(bo))
# solve(a == bo, u, bc, solver_parameters=sp)
unorm=sqrt(assemble(inner(grad(u), grad(u)) * dx))
uold=Function(V)
#uold.vector().axpy(1.0, u.vector())
uold.vector().axpy(1.0/unorm, u.vector())
#startnorm=assemble(inner(div(uold), div(uold)) * dx)
#print "startnorm=",startnorm

dlam=1.0
zs=0
iters = 0;
rip=10000.0
olam=0.0
zu = TrialFunction(V)
za_solver = LinearSolver(assemble(inner(grad(zu), grad(v))*dx + rip*div(zu)*div(v)*dx, bcs=bc), solver_parameters=sp)
while iters < max_iters or dlam > 1e-7:
# Scott-Vogelius iterated penalty method
    if iters % 10 == 0:
        ipmters = 0; ipmax_iters = 1000; div_u_norm = 1
        w  = Function(V)
        za = inner(grad(zu), grad(v))*dx + rip*div(zu)*div(v)*dx 
        while ipmters < ipmax_iters and div_u_norm > 1e-14:
        # solve and update w
            zu  = Function(V)
            bz = inner(grad(uold),grad(v))*dx - div(w)*div(v)*dx
            # solve(za == bz,zu,bc, solver_parameters=sp)
            za_solver.solve(zu, assemble(bz))
            w.vector().axpy(rip,zu.vector())
        # find the L^2 norm of div(u) to check stopping condition
            div_u_norm = sqrt(assemble(div(zu)*div(zu)*dx(mesh)))
            print("   IPM iter_no=",ipmters,"div_u_norm="," %.2e"%div_u_norm)
            ipmters += 1
        if ipmters >= ipmax_iters:
            print("IPM failed")
            import sys; sys.exit()
        znorm = sqrt(assemble(inner(grad(zu),grad(zu))*dx(mesh)))
        print("%   IPM iters=",ipmters,"div_z_norm="," %.2e"%div_u_norm)
        tenz=1000000000000.0*znorm
        if (tenz > unorm):
            zs+=1
            print("subtracting off the projection onto the div-free space")
            uold.vector().axpy(-1.0, zu.vector())
    b = div(uold)*div(v)*dx - mu*inner(grad(uold), grad(v))*dx
    # solve(a == b, u, bc, solver_parameters=sp)
    a_solver.solve(u, assemble(b))
    unorm=sqrt(assemble(inner(grad(u), grad(u)) * dx))
    dunrm=sqrt(assemble(inner(div(u), div(u)) * dx))
    lam=dunrm/unorm
    dlam=olam-lam
    olam=lam
    uold=Function(V)
    uold.vector().axpy(1.0/unorm, u.vector())
    znormrat=znorm/unorm
    print("% iter"," unorm","  div_unorm","  lambda", "   znormrat"," z restarts"," dlam")
    print("  ",iters," %.2e"%unorm," %.2e"%dunrm," %.4e"%lam," %.2e"%znormrat,"    ",zs,"    %.2e"%dlam)
    iters +=1

print("kin=",kin,"shift=","%.2e"%shft,"P=",kdeg," M=",meshsize,"its=",iters," lambda=","%.5e"%lam,domainstr," znr=","%.1e"%znormrat," zs=",zs," dlam=","%.2e"%dlam)

#plot(u, interactive=True)
#plot(div(u), interactive=True)

