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
| Soft-min robust LASSO with perturbed rows |  | `softmin_lasso` |

## Robust LASSO Quick Start

For robust sparse regression where each measurement has several possible
perturbed design rows, use `softmin_lasso`. Here `A_perturbed` has shape
`(m, K, n)`: `m` measurements, `K` candidate rows per measurement, and `n`
unknown coefficients.

```julia
m, K, n = 40, 5, 12
A_perturbed = randn(m, K, n)
y = randn(m)

lambda = 0.05
tau = 0.2

result = softmin_lasso(
    A_perturbed,
    y,
    lambda;
    tau,
    step=0.5,
    maxiter=1_000,
    return_info=true,
)

x = result.x
weights = result.weights  # soft row assignment weights, size (m, K)
```

`tau` controls how sharply the solver selects among perturbed rows. Smaller
values approach the hard objective `sum_i min_k residual[i, k]^2`, while larger
values average candidate rows more smoothly. You can pass a schedule to anneal
from smooth to sharp and warm-start each stage:

```julia
x = softmin_lasso(A_perturbed, y, lambda; tau=[1.0, 0.3, 0.1], step=0.5)
```

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

## Generic Parameter Sweeps

Use `parameter_sweep` when you want the same sweep logic for different solvers.
The sweep only needs a candidate list, a `solve` function, a scalar `score`, and
an optional stopping rule.

```julia
noise_l2 = 0.25
metric = result -> result.residual

sweep = parameter_sweep(
    [1e-2, 3e-3, 1e-3, 3e-4],
    lambda -> begin
        fit = nonnegative_lasso_fista(
            fista,
            lambda;
            warm_start=true,
            return_info=true,
        )
        residual = norm(A * fit.x - b)
        (lambda=lambda, x=copy(fit.x), fit=fit, residual=residual)
    end;
    stop=stop_when_below(metric, noise_l2),
    score=distance_score(metric, noise_l2),
)

lambda = sweep.result.lambda
x = sweep.result.x
sweep.trials
```

For monotone metrics that change roughly exponentially with the parameter,
`multiplicative_parameter_sweep` keeps the search multiplicative throughout. It
starts from one value, moves by `factor`, reverses direction whenever the target
is crossed, and replaces `factor` with `sqrt(factor)` after each crossing.

```julia
sweep = multiplicative_parameter_sweep(
    lambda -> begin
        fit = lasso_fista(A, b, lambda; return_info=true)
        residual = norm(A * fit.x - b)
        (lambda=lambda, x=copy(fit.x), fit=fit, residual=residual)
    end,
    1e-6;
    target=noise_l2,
    metric=result -> result.residual,
    factor=4.0,
    reltol=1e-3,
    steps=40,
)
```

By default the metric is assumed to increase with the parameter; set
`increasing=false` when larger parameter values reduce the metric. The returned
`sweep.stopped` tells you whether the relative/absolute target tolerance was
reached. `sweep.trials` records the candidate, metric, current factor, direction,
and whether each trial crossed the target.

Use `bracketed_parameter_sweep` when you want stricter bracket-then-bisect
behavior, or `bisection_parameter_sweep` directly when you already know good
`low` and `high` bounds. Warm starts are controlled inside the `solve` function,
not by the sweep engine. Create a reusable solver outside the closure, pass
`warm_start=true` after the first trial, and copy mutable outputs that should be
kept in `sweep.trials`.

```julia
fista = LassoFISTASolver(A, b)
trial = Ref(0)

sweep = multiplicative_parameter_sweep(
    lambda -> begin
        trial[] += 1
        fit = nonnegative_lasso_fista(
            fista,
            lambda;
            warm_start=trial[] > 1,
            return_info=true,
        )
        residual = norm(A * fit.x - b)
        (lambda=lambda, x=copy(fit.x), fit=fit, residual=residual)
    end,
    1e-6;
    target=noise_l2,
    metric=result -> result.residual,
)
```

Use the `snapshot` keyword when a solver returns mutable buffers that may be
changed by later warm-started trials.

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
same data and the L1 penalty shrinks active coefficients. The toolkit therefore
provides compute-heavier diagnostics that answer slightly different questions
instead of pretending there is one universal LASSO standard error.

Bootstrap resampling asks how much the answer changes when the observed rows are
resampled. Each bootstrap sample draws rows of `(A, b)` with replacement, solves
LASSO again, and summarizes the coefficient draws:

```julia
boot = nonnegative_lasso_bootstrap_uncertainty(
    A,
    b,
    lambda;
    samples=200,
    algorithm=:fista,
)

boot.coefficient_mean
boot.coefficient_std
boot.coefficient_interval.lower
boot.coefficient_interval.upper
boot.selection_probability
```

Use bootstrap when rows are reasonably exchangeable measurements and you want
coefficient intervals, selection probabilities, or prediction bands. It costs one
solve per sample.

Noise perturbation asks how much the answer changes under an assumed additive
Gaussian noise model. It repeatedly solves with `b + noise_sigma * randn(...)`.
If you know only the L2 norm of the noise, pass `noise_norm` instead:

```julia
noise_uncertainty = nonnegative_lasso_noise_perturbation_uncertainty(
    A,
    b,
    lambda;
    noise_norm=noise_l2,
    samples=200,
)
```

Use this when the noise model is more trustworthy than row resampling. It returns
the same summary fields as the bootstrap routine.

Stability selection asks which coefficients are reliably selected. It repeatedly
solves on random row subsets and reports how often each coefficient is active:

```julia
stability = nonnegative_lasso_stability_selection(
    A,
    b,
    [lambda, lambda / 3, lambda / 10];
    samples=200,
    subsample_fraction=0.5,
    threshold=0.8,
)

stability.selection_probability
stability.stable_support
```

Use stability selection when support confidence matters more than coefficient
intervals. A high selection probability means a component survives many data
subsets; it is not a classical p-value.

Lambda-path sensitivity asks whether conclusions depend on a narrow tuning
choice. It solves nearby lambdas with warm starts and records coefficient paths,
residual norms, L1 norms, and active-set indicators:

```julia
path = nonnegative_lasso_lambda_path(A, b, [1e-3, 3e-4, 1e-4])

path.coefficients
path.residuals
path.active
path.activation_min_lambda
path.activation_max_lambda
```

Use this to see whether components persist across regularization strengths or
appear only at one fragile lambda.

Refit asks what the coefficients look like after removing LASSO shrinkage
on the selected support. For signed models it refits ordinary least squares on
the active support. With `positive=true`, it refits a nonnegative least-squares
problem on that support and drops selected coefficients that cannot stay positive:

```julia
x_lasso = nonnegative_lasso_fista(A, b, lambda)
refit = lasso_refit(A, b, x_lasso; positive=true)

refit.x
refit.support
refit.original_support
refit.standard_error
refit.covariance
```

The refit standard errors are conditional on the final support. They are useful
for diagnostics, but they do not include support-selection uncertainty; combine
with bootstrap or stability selection when support uncertainty matters.

Debiased LASSO applies an analytic correction to the LASSO estimate:

```julia
deb = debiased_lasso(A, b, lambda; noise_norm=noise_l2)

x_debiased = deb.x
deb.standard_error
deb.covariance
deb.bias_operator_infnorm
```

The correction is `x_hat + (1 / m) * M * A' * (b - A * x_hat)`, where `m` is the
number of measurements. By default `M = pinv(A' * A / m)`, but you can pass a
custom approximate inverse with `M=...`. The reported covariance is the Gaussian
term `sigma^2 * M * A' * A * M' / m^2`; `sigma^2` comes from `noise_variance`,
`noise_sigma`, `noise_norm`, or a residual estimate. The field
`bias_operator_infnorm` summarizes how close `M * (A' * A / m)` is to identity;
smaller is better for the theoretical remainder term.

Cross-validation results also include `validation_standard_errors` and
`lambda_1se`. The one-standard-error rule picks the largest lambda whose CV error
is within one standard error of the best CV error, giving a more conservative
sparser model when several lambdas perform similarly.


## Useful Options

All solver calls accept stopping options. With `return_info=true`, they return
a diagnostics result:

```julia
result = nonnegative_lasso_fista(
    A,
    b,
    lambda;
    abstol=1e-7,
    reltol=1e-5,
    maxiter=20_000,
    return_info=true,
)

x = result.x
result.converged
result.iterations
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
   result = nonnegative_lasso_fista(
       fista,
       lambda;
       maxiter=2_000,
       return_info=true,
   )
   ```

Check `result.converged`, `result.iterations`, and your reconstruction or
residual quality before treating a capped solve as final.

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
