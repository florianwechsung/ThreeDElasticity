from firedrake import *
import argparse
from bary import BaryMeshHierarchy
import numpy


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--dim", type=int, default=2)
parser.add_argument("--baseN", type=int, default=4)
parser.add_argument("--nref", type=int, default=1)
parser.add_argument("--k", type=int, default=4)
parser.add_argument("--mh", type=str, default="uniform", choices=["bary", "uniformbary", "uniform"])
args, _ = parser.parse_known_args()

distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}

N = args.baseN
nref = args.nref
k = args.k
gamma = Constant(1)
dim = args.dim


warning("Let's make some meshes.")

if dim == 2:
    base = UnitSquareMesh(N, N, distribution_parameters=distribution_parameters)#, diagonal="crossed")
    # base = Mesh("mesh2d.msh", distribution_parameters=distribution_parameters)
elif dim == 3:
    base = UnitCubeMesh(N, N, N, distribution_parameters=distribution_parameters)
    # base = Mesh("cube-1.msh", distribution_parameters=distribution_parameters)

if args.mh == "bary":
    mh = BaryMeshHierarchy(base, nref,
                           reorder=True, distribution_parameters=distribution_parameters)
elif args.mh == "uniformbary":
    bmesh = Mesh(bary(base._plex), distribution_parameters={"partition": False})
    mh = MeshHierarchy(bmesh, nref, reorder=True,
                       distribution_parameters=distribution_parameters)
elif args.mh == "uniform":
    mh = MeshHierarchy(base, nref, reorder=True,
                       distribution_parameters=distribution_parameters)
else:
    raise NotImplementedError("Only know bary, uniformbary and uniform for the hierarchy.")

warning("Meshes are ready.")

V = VectorFunctionSpace(mh[-1], "CG", k)
warning("Function space ready.")

u = Function(V)
v = TestFunction(V)

X = SpatialCoordinate(mh[-1])
n = FacetNormal(mh[-1])
if dim == 2:
    x, y = X
    # u_ex = as_vector([0, -x**2])
    u_ex = as_vector([0, -sin(x)**2])
    # f = as_vector([0, sin(X[0])*exp(X[1])])
    bclabels = [1]
else:
    x, y, z = X
    u_ex = as_vector([0, 0, -sin(x)**2])
    f = -div(grad(u_ex)) - gamma * grad(div(u_ex))
    # f = as_vector([0, sin(X[0])*exp(X[1]), 0])
    bclabels = [1]

f = -div(grad(u_ex)) - gamma * grad(div(u_ex))
g = dot(grad(u_ex), n) + gamma * div(u_ex)*n
F = inner(grad(u), grad(v))*dx + gamma*inner(div(u), div(v))*dx - inner(f, v)*dx - inner(g, v) * ds

warning("Creating Dirichlet boundary condition.")
bcs = DirichletBC(V, u_ex, bclabels)
warning("Boundary condition ready.")
# from estimate_inf_sup import estimate_inf_sup
# estimate_inf_sup(V, bcs)
# import sys
# sys.exit()
sp = {
       "mat_type": "matfree",
       "pmat_type": "aij",
       "snes_type": "ksponly",
       "ksp_type": "gmres",
       "ksp_rtol": 1.0e-13,
       "ksp_atol": 0.0,
       "ksp_max_it": 1000,
       "ksp_monitor_true_residual": None,
       "ksp_converged_reason": None,
       "ksp_norm_type": "unpreconditioned",
       # "pc_type": "lu",
       # "pc_factor_mat_solver_type": "superlu_dist",
       "pc_type": "mg",
       "mg_coarse_ksp_type": "preonly",
       "mg_coarse_pc_type": "python",
       "mg_coarse_pc_python_type": "firedrake.AssembledPC",
       "mg_coarse_assembled_pc_type": "lu",
       "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
       "mg_levels_ksp_type": "richardson",
       "mg_levels_ksp_max_it": 1,
       "mg_levels_ksp_richardson_scale": 1/(dim+1),
       "mg_levels_pc_type": "python",
       "mg_levels_pc_python_type": "firedrake.PatchPC",
       "mg_levels_patch_pc_patch_save_operators": True,
       "mg_levels_patch_pc_patch_partition_of_unity": False,
       "mg_levels_patch_pc_patch_sub_mat_type": "seqaij",
       # "mg_levels_patch_pc_patch_sub_mat_type": "dense",
       "mg_levels_patch_pc_patch_multiplicative": False,
       "mg_levels_patch_pc_patch_symmetrise_sweep": False,
       "mg_levels_patch_pc_patch_construct_dim": 0,
       "mg_levels_patch_pc_patch_construct_type":  "star",
       # "mg_levels_patch_pc_patch_dense_inverse":  True,
       "mg_levels_patch_sub_ksp_type": "preonly",
       "mg_levels_patch_sub_pc_type": "lu",
     }

pvd = File("output/u_ex-dim-%i-k-%i-%s-unstructured.pvd" % (args.dim, args.k, args.mh))
pvd.write(u.interpolate(u_ex))
pvd = File("output/output-dim-%i-k-%i-%s-unstructured.pvd" % (args.dim, args.k, args.mh))
u.rename("Solution")

# for gamma_ in [10000000]:
for gamma_ in [0., 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]:
    gamma.assign(gamma_)
    u.assign(0)
    warning("Launching solve for gamma = %s." % gamma_)
    solve(F == 0, u, bcs, solver_parameters=sp)
    warning("Solve finished. Norm of solution = %s." % norm(u))
    pvd.write(u, time=gamma_)
    warning("||div(u)||=%f" % norm(div(u)))
    warning("||u-u_ex||=%e" % norm(u-u_ex))
