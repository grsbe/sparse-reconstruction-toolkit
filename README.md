# Sparse Reconstruction Toolkit

`SparseReconstructionToolkit` is a Julia toolkit for sparse linear inverse problems of the form

```math
\min_x \frac{1}{2}\|Ax+Bz-b\|_2^2 + \lambda\|x\|_1 + \eta\|z\|_2^2
```

and related L1-ball constrained variants. It currently provides ADMM and FISTA
solvers for standard and nonnegative LASSO problems, a two-block LASSO-ridge
FISTA solver, plus Gurobi helper solvers and benchmark scripts for comparison.

The solvers use `ProximalOperators.jl` for the L1, nonnegative, and L1-ball
proximal steps.

## Quick Start

Install `SparseReconstructionToolkit` directly from GitHub in a Julia environment:

```julia
using Pkg
Pkg.add(url="https://github.com/grsbe/sparse-reconstruction-toolkit.git")
```

Then load it like any other package:

```julia
using SparseReconstructionToolkit
using LinearAlgebra

A = Matrix{Float64}(I, 3, 3)
b = [2.0, -1.0, 0.2]
lambda = 0.5

x_admm = lasso_admm(A, b, lambda)
x_fista = lasso_fista(A, b, lambda)
x_positive = nonnegative_lasso_fista(A, b, lambda)
```

For a development checkout, start Julia from the repository root with
`julia --project=.` and use the same `using SparseReconstructionToolkit` import.

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
| L1-regularized `x` and ridge-regularized `z` |  | `lasso_ridge_fista` |
| Nonnegative L1-regularized `x` and ridge-regularized `z` |  | `nonnegative_lasso_ridge_fista` |

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

For a sparse block `x` and ridge block `z`, use the LASSO-ridge FISTA solver:

```julia
result = lasso_ridge_fista(A, B, b, lambda_x, lambda_z)
x = result.x
z = result.z
```

If both blocks should be nonnegative, use the dedicated wrapper:

```julia
positive_result = nonnegative_lasso_ridge_fista(A, B, b, lambda_x, lambda_z)
x = positive_result.x
z = positive_result.z
```

It minimizes

```math
\frac{1}{2}\|Ax+Bz-b\|_2^2 + \lambda_x\|x\|_1 +
\frac{\lambda_z}{2}\|z\|_2^2.
```

Use `x_nonnegative=true` or `z_nonnegative=true` to constrain either block to
be nonnegative, or call `nonnegative_lasso_ridge_fista` when both blocks should
be constrained. Reusable `LassoRidgeFISTASolver` and `LassoRidgeFISTAWorkspace`
objects follow the ordinary FISTA API.

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

If the L2 norm of the noise is known, the package can choose `lambda` by a
warm-started residual-matching search:

```julia
result = nonnegative_lasso_noise_sweep(
    A,
    b,
    noise_l2;
    algorithm=:fista,
    steps=12,
    step=fista_step_size(A; method=:power),
)

lambda = result.lambda
x = result.x
result.residual
```

Use `lasso_noise_sweep` for signed coefficients or set `algorithm=:admm` to run
the same bisection with ADMM.

You can also estimate `lambda` with row-wise k-fold cross-validation:

```julia
cv = nonnegative_lasso_crossvalidation(
    A,
    b;
    folds=5,
    lambdas=[1e-3, 3e-4, 1e-4, 3e-5],
    algorithm=:fista,
)

lambda = cv.lambda
x = cv.x
cv.validation_error
```

Use `folds=3` for a lighter 3-way sweep. If `lambdas` is omitted, a default
geometric grid is built from the data. CV results also include fold standard
errors and `lambda_1se` for the one-standard-error rule.

## Uncertainty Diagnostics

LASSO uncertainty is approximate because the active set is selected from the
same data. The toolkit provides compute-heavier diagnostics for sensitivity and
support stability:

```julia
boot = nonnegative_lasso_bootstrap_uncertainty(A, b, lambda; samples=200)
stability = nonnegative_lasso_stability_selection(A, b, lambda; samples=200)
path = nonnegative_lasso_lambda_path(A, b, [1e-3, 3e-4, 1e-4])
refit = debiased_lasso_refit(A, b, boot.coefficient_mean; positive=true)
```

Use `lasso_noise_perturbation_uncertainty` when you have an assumed noise
standard deviation or L2 noise norm. Bootstrap and perturbation summaries return
coefficient intervals, coefficient standard deviations, selection probabilities,
and prediction means/standard deviations.


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

FISTA computes an exact spectral-norm step size when a workspace is created.
For a large forward matrix `A`, or a matrix-free forward operator used in its
place, computing that exact norm can be expensive. Use the power-iteration
helper to get a much cheaper estimated step size:

```julia
step = fista_step_size(A; method=:power)
fista = LassoFISTASolver(A, b; step)
```

The power estimate uses `iterations=20` and a `safety=1.05` margin by default.
Raise either value if the operator is difficult to estimate. Avoid
`1 / norm(A)^2` for a dense matrix unless the conservative Frobenius-norm bound
is intentional; it can make FISTA take much smaller steps than necessary.

## Performance Tips

The first knobs to try on repeated or large inversion problems are:

1. Reuse a solver and warm start nearby parameter sweeps:

   ```julia
   fista = LassoFISTASolver(A, b; step=fista_step_size(A; method=:power))
   x1 = nonnegative_lasso_fista(fista, 1e-5)
   x2 = nonnegative_lasso_fista(fista, 5e-6; warm_start=true)
   ```

2. Use a power-iteration FISTA step estimate when the exact spectral norm of
   `A` is costly:

   ```julia
   step = fista_step_size(A; method=:power)
   ```

3. Restrict `maxiter` when an approximate answer is enough, especially inside
   a lambda search:

   ```julia
   info = nonnegative_lasso_fista(
       fista,
       lambda;
       maxiter=2_000,
       return_info=true,
   )
   ```

Check `info.converged`, `info.iterations`, and your reconstruction or residual
quality before treating a capped solve as final.

## Threading

The ADMM/FISTA iteration order is sequential, but dense linear algebra inside
each iteration can use BLAS threads. This matters for large dense operators.

```julia
using LinearAlgebra
BLAS.set_num_threads(32)
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
the package:

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
julia --project=. benchmark/testdata_nonnegative_lasso_ridge_fista.jl
```

`testdata_positive_inversion.jl` compares positive Gurobi BPDN against
positive radius-constrained ADMM and FISTA.

`testdata_positive_penalized_search.jl` searches the positive penalized LASSO
hyperparameter `lambda` until the residual norm is close to the supplied noise
norm.

`testdata_nonnegative_lasso_ridge_fista.jl` times a sparse-plus-smooth toy
problem built from the test-data forward operator with 32 BLAS threads and a
power-iteration FISTA step size, and uses `nonnegative_lasso_ridge_fista` for
the solver comparison.


## Notes

- ADMM can be very fast when its reusable least-squares state is amortized
  over many solves.
- FISTA is attractive for large dense operators because its iterations are
  matrix-vector products plus cheap proximal steps.
- The best algorithm and hyperparameters depend on operator scaling, noise,
  stopping tolerances, and whether many related solves are performed.
