using LinearAlgebra
using Random
using Statistics
using SparseReconstructionToolkit

const SAMPLE_COUNT = 5
const N_SAMPLES = 500
const N_FEATURES = 100
const N_NONZERO = 12

function make_problem(; seed=20260521, nonnegative=false)
    rng = MersenneTwister(seed)
    A = randn(rng, N_SAMPLES, N_FEATURES) / sqrt(N_SAMPLES)
    x_true = zeros(N_FEATURES)
    support = randperm(rng, N_FEATURES)[1:N_NONZERO]
    coefficients = nonnegative ? rand(rng, N_NONZERO) : randn(rng, N_NONZERO)
    x_true[support] .= coefficients
    b = A * x_true + 0.01 * randn(rng, N_SAMPLES)
    lambda = 0.05 * norm(transpose(A) * b, Inf)
    return A, b, lambda
end

lasso_objective(A, b, lambda, x) =
    0.5 * norm(A * x - b)^2 + lambda * norm(x, 1)

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

function benchmark_lasso(; nonnegative=false)
    A, b, lambda = make_problem(; nonnegative)
    admm_solver = LassoADMMSolver(A, b; rho=1.0)
    fista_solver = LassoFISTASolver(A, b)
    solve_admm = nonnegative ? nonnegative_lasso_admm : lasso_admm
    solve_fista = nonnegative ? nonnegative_lasso_fista : lasso_fista

    # Warm up compiled paths and the solver caches before timing.
    solve_admm(A, b, lambda; rho=1.0, abstol=1e-6, reltol=1e-4)
    solve_admm(admm_solver, lambda; abstol=1e-6, reltol=1e-4)
    solve_fista(A, b, lambda; abstol=1e-6, reltol=1e-4)
    solve_fista(fista_solver, lambda; abstol=1e-6, reltol=1e-4)

    x_admm, t_admm, admm_samples = timed_median(
        () -> solve_admm(A, b, lambda; rho=1.0, abstol=1e-6, reltol=1e-4),
    )
    x_admm_solver, t_admm_solver, admm_solver_samples = timed_median(
        () -> solve_admm(admm_solver, lambda; abstol=1e-6, reltol=1e-4),
    )
    x_fista, t_fista, fista_samples = timed_median(
        () -> solve_fista(A, b, lambda; abstol=1e-6, reltol=1e-4),
    )
    x_fista_solver, t_fista_solver, fista_solver_samples = timed_median(
        () -> solve_fista(fista_solver, lambda; abstol=1e-6, reltol=1e-4),
    )

    admm_bytes = @allocated solve_admm(
        A,
        b,
        lambda;
        rho=1.0,
        abstol=1e-6,
        reltol=1e-4,
    )
    admm_solver_bytes = @allocated solve_admm(
        admm_solver,
        lambda;
        abstol=1e-6,
        reltol=1e-4,
    )
    fista_bytes = @allocated solve_fista(A, b, lambda; abstol=1e-6, reltol=1e-4)
    fista_solver_bytes = @allocated solve_fista(
        fista_solver,
        lambda;
        abstol=1e-6,
        reltol=1e-4,
    )

    objective_admm = lasso_objective(A, b, lambda, x_admm)
    objective_admm_solver = lasso_objective(A, b, lambda, x_admm_solver)
    objective_fista = lasso_objective(A, b, lambda, x_fista)
    objective_fista_solver = lasso_objective(A, b, lambda, x_fista_solver)
    reference_objective = min(
        objective_admm,
        objective_admm_solver,
        objective_fista,
        objective_fista_solver,
    )

    title = nonnegative ? "Nonnegative LASSO FISTA versus ADMM" : "LASSO FISTA versus ADMM"
    println(title)
    println("samples=$(N_SAMPLES), features=$(N_FEATURES), true_nonzeros=$(N_NONZERO)")
    println("lambda=$(lambda), BLAS_threads=$(BLAS.get_num_threads())")
    println()
    print_result("ADMM one-shot", t_admm, admm_samples, admm_bytes, objective_admm)
    print_result(
        "ADMM reused solver",
        t_admm_solver,
        admm_solver_samples,
        admm_solver_bytes,
        objective_admm_solver,
    )
    print_result("FISTA one-shot", t_fista, fista_samples, fista_bytes, objective_fista)
    print_result(
        "FISTA reused solver",
        t_fista_solver,
        fista_solver_samples,
        fista_solver_bytes,
        objective_fista_solver,
    )
    println("FISTA one-shot relative objective gap: $(abs(objective_fista - objective_admm_solver) / max(reference_objective, eps()))")
    println("FISTA reused solver relative objective gap: $(abs(objective_fista_solver - objective_admm_solver) / max(reference_objective, eps()))")
    println("FISTA reused solver coefficient relative difference: $(norm(x_fista_solver - x_admm_solver) / max(norm(x_admm_solver), eps()))")
    nonnegative && println("FISTA minimum coefficient: $(minimum(x_fista_solver))")
    println()
end

function main()
    BLAS.set_num_threads(1)
    benchmark_lasso()
    benchmark_lasso(; nonnegative=true)
end

main()
