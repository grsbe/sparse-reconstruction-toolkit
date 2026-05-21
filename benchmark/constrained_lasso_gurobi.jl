using LinearAlgebra
using Random
using Statistics
using srt

include(joinpath(@__DIR__, "..", "src", "gurobi_constrained_lasso_solver.jl"))

const SAMPLE_COUNT = 3
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
    radius = 0.6 * norm(x_true, 1)
    return A, b, radius
end

least_squares_objective(A, b, x) = 0.5 * norm(A * x - b)^2

function solve_srt(A, b, radius)
    return constrained_lasso_admm(
        A,
        b,
        radius;
        rho=1.0,
        abstol=1e-6,
        reltol=1e-4,
    )
end

function solve_srt(solver, radius)
    return constrained_lasso_admm(
        solver,
        radius;
        abstol=1e-6,
        reltol=1e-4,
    )
end

solve_gurobi(A, b, radius) = constrained_lasso_gurobi(A, b, radius)

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

function print_result(name, seconds, samples, bytes, objective, l1_norm, radius)
    println("$(name) median seconds: $(seconds)")
    println("$(name) samples seconds: $(samples)")
    println("$(name) allocations bytes: $(bytes)")
    println("$(name) objective: $(objective)")
    println("$(name) l1 norm: $(l1_norm)")
    println("$(name) radius violation: $(max(l1_norm - radius, 0.0))")
    println()
end

function main()
    BLAS.set_num_threads(1)
    A, b, radius = make_problem()
    solver = LassoADMMSolver(A, b; rho=1.0)

    # Warm up JIT compilation, ADMM factorization, and the Gurobi model path.
    solve_srt(A, b, radius)
    solve_srt(solver, radius)
    solve_gurobi(A, b, radius)

    x_srt, t_srt, srt_samples = timed_median(() -> solve_srt(A, b, radius))
    x_solver, t_solver, solver_samples =
        timed_median(() -> solve_srt(solver, radius))
    x_gurobi, t_gurobi, gurobi_samples =
        timed_median(() -> solve_gurobi(A, b, radius))

    srt_bytes = @allocated solve_srt(A, b, radius)
    solver_bytes = @allocated solve_srt(solver, radius)
    gurobi_bytes = @allocated solve_gurobi(A, b, radius)

    objective_srt = least_squares_objective(A, b, x_srt)
    objective_solver = least_squares_objective(A, b, x_solver)
    objective_gurobi = least_squares_objective(A, b, x_gurobi)
    reference_objective = min(objective_srt, objective_solver, objective_gurobi)

    println("Constrained LASSO benchmark")
    println("samples=$(N_SAMPLES), features=$(N_FEATURES), true_nonzeros=$(N_NONZERO)")
    println("radius=$(radius), BLAS_threads=$(BLAS.get_num_threads())")
    println()
    print_result(
        "srt one-shot ADMM",
        t_srt,
        srt_samples,
        srt_bytes,
        objective_srt,
        norm(x_srt, 1),
        radius,
    )
    print_result(
        "srt reused solver ADMM",
        t_solver,
        solver_samples,
        solver_bytes,
        objective_solver,
        norm(x_solver, 1),
        radius,
    )
    print_result(
        "Gurobi constrained LASSO",
        t_gurobi,
        gurobi_samples,
        gurobi_bytes,
        objective_gurobi,
        norm(x_gurobi, 1),
        radius,
    )
    println("srt one-shot relative objective gap: $(abs(objective_srt - objective_gurobi) / max(reference_objective, eps()))")
    println("srt solver relative objective gap: $(abs(objective_solver - objective_gurobi) / max(reference_objective, eps()))")
    println("reused solver coefficient relative difference: $(norm(x_solver - x_gurobi) / max(norm(x_gurobi), eps()))")
end

main()
