using JLD2
using LinearAlgebra
using Printf
using Statistics
using srt

include(joinpath(@__DIR__, "..", "src", "gurobi_constrained_lasso_solver.jl"))

const DATA_DIR = joinpath(@__DIR__, "..", "testdata")
const BLAS_THREADS = 32
const FISTA_SAMPLES = 3
const GUROBI_SAMPLES = 1
const SNR30_LAMBDA = 2.84e-7
const SOLVER_ABSTOL = 1e-7
const SOLVER_RELTOL = 1e-5
const SOLVER_MAXITER = 2_000

load_single(name) = JLD2.load(joinpath(DATA_DIR, name))["single_stored_object"]
lasso_objective(A, y, lambda, x) =
    0.5 * norm(A * x - y)^2 + lambda * norm(x, 1)

function timed_samples(f; samples)
    times = Vector{Float64}(undef, samples)
    result = nothing
    for sample in eachindex(times)
        t0 = time_ns()
        result = f()
        times[sample] = (time_ns() - t0) / 1e9
    end
    return result, median(times), times
end

function report(name, A, y, x_true, noise, lambda, x, seconds, samples)
    residual = norm(A * x - y)
    relative_error = norm(x - x_true) / max(norm(x_true), eps())
    @printf(
        "%-26s median=%9.3f s  samples=%s\n",
        name,
        seconds,
        string(samples),
    )
    @printf(
        "  objective=%9.5g residual=%9.5g residual-noise=%9.3g rel_x_err=%9.4g l1=%9.4g min=%9.3g\n",
        lasso_objective(A, y, lambda, x),
        residual,
        residual - noise,
        relative_error,
        norm(x, 1),
        minimum(x),
    )
end

function main()
    BLAS.set_num_threads(BLAS_THREADS)
    A = load_single("A.jld2")
    y = load_single("y_noisy_SNR30.jld2")
    noise = load_single("L2_norm_of_noise_SNR30.jld2")
    x_true = vec(permutedims(load_single("target.jld2")))
    step = fista_step_size(A; method=:power)

    println("SNR 30 positive penalized LASSO benchmark")
    println("A_size=$(size(A)), lambda=$(SNR30_LAMBDA), noise=$(noise)")
    println("BLAS_threads=$(BLAS.get_num_threads()), FISTA_step=$(step)")
    println()

    fista_solver, fista_setup_seconds =
        timed_samples(() -> LassoFISTASolver(A, y; step); samples=1)
    fista_info, fista_seconds, fista_samples = timed_samples(
        () -> nonnegative_lasso_fista(
            fista_solver,
            SNR30_LAMBDA;
            abstol=SOLVER_ABSTOL,
            reltol=SOLVER_RELTOL,
            maxiter=SOLVER_MAXITER,
            return_info=true,
        );
        samples=FISTA_SAMPLES,
    )

    # Warm up the Gurobi model path on a tiny positive LASSO before timing.
    tiny_A = Matrix{Float64}(I, 3, 3)
    lasso_gurobi(tiny_A, ones(3), 0.1; positive=true)
    x_gurobi, gurobi_seconds, gurobi_samples = timed_samples(
        () -> lasso_gurobi(A, y, SNR30_LAMBDA; positive=true);
        samples=GUROBI_SAMPLES,
    )

    @printf("FISTA solver setup=%0.3f s\n", fista_setup_seconds)
    @printf(
        "FISTA termination: iterations=%d converged=%s iterate_change=%9.3g\n",
        fista_info.iterations,
        string(fista_info.converged),
        fista_info.iterate_change,
    )
    report(
        "positive FISTA LASSO",
        A,
        y,
        x_true,
        noise,
        SNR30_LAMBDA,
        fista_info.x,
        fista_seconds,
        fista_samples,
    )
    report(
        "positive Gurobi LASSO",
        A,
        y,
        x_true,
        noise,
        SNR30_LAMBDA,
        x_gurobi,
        gurobi_seconds,
        gurobi_samples,
    )
    @printf(
        "coefficient relative difference=%9.4g\n",
        norm(fista_info.x - x_gurobi) / max(norm(x_gurobi), eps()),
    )
end

main()
