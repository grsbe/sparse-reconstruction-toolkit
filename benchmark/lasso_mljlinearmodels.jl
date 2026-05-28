using LinearAlgebra
using MLJLinearModels
using Random
using Statistics
using SparseReconstructionToolkit

const SAMPLE_COUNT = 5
const N_SAMPLES = 500
const N_FEATURES = 100
const N_NONZERO = 12

function make_problem(; seed=20260521)
    rng = MersenneTwister(seed)
    A = randn(rng, N_SAMPLES, N_FEATURES) / sqrt(N_SAMPLES)
    x_true = zeros(N_FEATURES)
    support = randperm(rng, N_FEATURES)[1:N_NONZERO]
    x_true[support] .= randn(rng, N_NONZERO)
    b = A * x_true + 0.01 * randn(rng, N_SAMPLES)
    lambda = 0.05 * norm(A' * b, Inf)
    return A, b, lambda
end

lasso_objective(A, b, lambda, x) = 0.5 * norm(A * x - b)^2 + lambda * norm(x, 1)

function solve_srt(A, b, lambda)
    return lasso_admm(A, b, lambda; rho=1.0, abstol=1e-6, reltol=1e-4)
end

function solve_srt(solver, lambda)
    return lasso_admm(solver, lambda; abstol=1e-6, reltol=1e-4)
end

function solve_mlj(A, b, lambda)
    lasso = LassoRegression(
        lambda;
        fit_intercept=false,
        scale_penalty_with_samples=false,
    )
    return fit(lasso, A, b; solver=FISTA(tol=1e-4, max_iter=10_000))
end

function timed_median(f; samples=SAMPLE_COUNT)
    times = Vector{Float64}(undef, samples)
    result = nothing

    for sample in eachindex(times)
        t0 = time_ns()
        result = f()
        times[sample] = (time_ns() - t0) / 1e9
    end

    return result, median(times), times
end

function print_result(name, seconds, samples, bytes, objective)
    println("$(name) median seconds: $(seconds)")
    println("$(name) samples seconds: $(samples)")
    println("$(name) allocations bytes: $(bytes)")
    println("$(name) objective: $(objective)")
    println()
end

function main()
    BLAS.set_num_threads(1)
    A, b, lambda = make_problem()
    solver = LassoADMMSolver(A, b; rho=1.0)

    # Warm up JIT compilation, ADMM factorization, and the solver paths.
    solve_srt(A, b, lambda)
    solve_srt(solver, lambda)
    solve_mlj(A, b, lambda)

    x_srt, t_srt, srt_samples = timed_median(() -> solve_srt(A, b, lambda))
    x_solver, t_solver, solver_samples =
        timed_median(() -> solve_srt(solver, lambda))
    x_mlj, t_mlj, mlj_samples = timed_median(() -> solve_mlj(A, b, lambda))

    srt_bytes = @allocated solve_srt(A, b, lambda)
    solver_bytes = @allocated solve_srt(solver, lambda)
    mlj_bytes = @allocated solve_mlj(A, b, lambda)

    objective_srt = lasso_objective(A, b, lambda, x_srt)
    objective_solver = lasso_objective(A, b, lambda, x_solver)
    objective_mlj = lasso_objective(A, b, lambda, x_mlj)
    reference_objective = min(objective_srt, objective_solver, objective_mlj)

    println("Unconstrained LASSO benchmark")
    println("samples=$(N_SAMPLES), features=$(N_FEATURES), true_nonzeros=$(N_NONZERO)")
    println("lambda=$(lambda), BLAS_threads=$(BLAS.get_num_threads())")
    println()
    print_result("srt one-shot ADMM", t_srt, srt_samples, srt_bytes, objective_srt)
    print_result(
        "srt reused solver ADMM",
        t_solver,
        solver_samples,
        solver_bytes,
        objective_solver,
    )
    print_result("MLJLinearModels FISTA", t_mlj, mlj_samples, mlj_bytes, objective_mlj)
    println("srt one-shot relative objective gap: $(abs(objective_srt - objective_mlj) / max(reference_objective, eps()))")
    println("srt solver relative objective gap: $(abs(objective_solver - objective_mlj) / max(reference_objective, eps()))")
    println("reused solver coefficient relative difference: $(norm(x_solver - x_mlj) / max(norm(x_mlj), eps()))")
end

main()
