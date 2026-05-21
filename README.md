# Sparse Reconstruction Toolkit

`srt` is a Julia toolkit for sparse linear inverse problems of the form

```math
\min_x \frac{1}{2}\|Ax-b\|_2^2 + \lambda\|x\|_1
```

and related L1-ball constrained variants. It currently provides ADMM and FISTA
solvers for standard and nonnegative LASSO problems, plus Gurobi helper solvers
and benchmark scripts for comparison.

The solvers use `ProximalOperators.jl` for the L1, nonnegative, and L1-ball
proximal steps.

## Quick Start

Run commands from the repository root:

```bash
cd /home/tobias-grasberger/sparse-reconstruction-toolkit
julia --project=.
```

Then in Julia:

```julia
using srt
using LinearAlgebra

A = Matrix{Float64}(I, 3, 3)
b = [2.0, -1.0, 0.2]
lambda = 0.5

x_admm = lasso_admm(A, b, lambda)
x_fista = lasso_fista(A, b, lambda)
x_positive = nonnegative_lasso_fista(A, b, lambda)
```

Run the package tests with:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Solver Choices

| Problem | ADMM | FISTA |
| --- | --- | --- |
| Penalized LASSO | `lasso_admm` | `lasso_fista` |
| Penalized LASSO with `x >= 0` | `nonnegative_lasso_admm` | `nonnegative_lasso_fista` |
| L1-ball constrained least squares | `constrained_lasso_admm` | `constrained_lasso_fista` |
| Nonnegative L1-ball constrained least squares | `nonnegative_constrained_lasso_admm` | `nonnegative_constrained_lasso_fista` |

The penalized calls use `lambda`:

```julia
x = nonnegative_lasso_admm(A, b, lambda)
```

The constrained calls use an L1 radius:

```julia
radius = 1.0
x = nonnegative_constrained_lasso_fista(A, b, radius)
```

`nonnegative` means `x >= 0`. Zeros are still allowed and are important for
sparse reconstructions.

## Repeated Solves

One-shot functions build solver state for a single call. For repeated solves
with the same `A` and `b`, create a reusable solver object.

ADMM:

```julia
admm = LassoADMMSolver(A, b; rho=1.0)

x1 = lasso_admm(admm, 0.2)
x2 = nonnegative_lasso_admm(admm, 0.1)
x3 = nonnegative_constrained_lasso_admm(admm, 1.0)
```

FISTA:

```julia
fista = LassoFISTASolver(A, b)

x1 = lasso_fista(fista, 0.2)
x2 = nonnegative_lasso_fista(fista, 0.1)
x3 = nonnegative_constrained_lasso_fista(fista, 1.0)
```

Solver objects are mutable and reuse internal solution buffers. Copy a result
before the next solve if it must be kept:

```julia
x_saved = copy(nonnegative_lasso_fista(fista, 0.1))
```

For nearby parameter sweeps, warm starts can help:

```julia
x1 = nonnegative_lasso_fista(fista, 1e-5)
x2 = nonnegative_lasso_fista(fista, 5e-6; warm_start=true)
```

The lower-level `LassoADMMWorkspace` and `LassoFISTAWorkspace` APIs remain
available when direct control over cached state is useful.

## Useful Options

All solver calls accept stopping options such as:

```julia
info = nonnegative_lasso_fista(
    A,
    b,
    lambda;
    abstol=1e-7,
    reltol=1e-5,
    maxiter=20_000,
    return_info=true,
)

x = info.x
info.converged
info.iterations
```

ADMM also accepts `rho` when creating one-shot or reusable ADMM solvers:

```julia
x = nonnegative_lasso_admm(A, b, lambda; rho=0.1)
admm = LassoADMMSolver(A, b; rho=0.1)
```

`rho` does not change the LASSO problem. It changes ADMM convergence behavior.

FISTA computes a default step size when a workspace is created. A custom step
can be supplied when a known safe step size is available:

```julia
fista = LassoFISTASolver(A, b; step=1 / norm(A)^2)
```

## Threading

The ADMM/FISTA iteration order is sequential, but dense linear algebra inside
each iteration can use BLAS threads. This matters for large dense operators.

```julia
using LinearAlgebra
BLAS.set_num_threads(8)
BLAS.get_num_threads()
```

Benchmark the thread count for the target machine and problem size. Small
problems may be faster with one BLAS thread; the included `testdata` operator
benefited from more BLAS threads in local measurements.

## Gurobi Reference Helpers

`src/gurobi_constrained_lasso_solver.jl` contains comparison/reference helpers
built with JuMP and Gurobi:

- `basis_pursuit_denoising_gurobi(A, b; noise, positive=false)`
- `lasso_gurobi(A, b, lambda; positive=false)`
- `constrained_lasso_gurobi(A, b, radius; positive=false)`

These helpers are included by the benchmark scripts rather than exported from
`srt`:

```julia
include("src/gurobi_constrained_lasso_solver.jl")

x = basis_pursuit_denoising_gurobi(A, b; noise=0.01, positive=true)
```

Gurobi needs a working Gurobi installation and license.

## Benchmarks

Run a benchmark script from the repo root:

```bash
julia --project=. benchmark/lasso_fista_admm.jl
julia --project=. benchmark/lasso_mljlinearmodels.jl
julia --project=. benchmark/constrained_lasso_gurobi.jl
julia --project=. benchmark/nonnegative_lasso_gurobi.jl
```

Real-data inversion comparisons are also included:

```bash
julia --project=. benchmark/testdata_positive_inversion.jl
julia --project=. benchmark/testdata_positive_penalized_search.jl
```

`testdata_positive_inversion.jl` compares positive Gurobi BPDN against
positive radius-constrained ADMM and FISTA.

`testdata_positive_penalized_search.jl` searches the positive penalized LASSO
hyperparameter `lambda` until the residual norm is close to the supplied noise
norm.


## Notes

- ADMM can be very fast when its reusable least-squares state is amortized
  over many solves.
- FISTA is attractive for large dense operators because its iterations are
  matrix-vector products plus cheap proximal steps.
- The best algorithm and hyperparameters depend on operator scaling, noise,
  stopping tolerances, and whether many related solves are performed.
