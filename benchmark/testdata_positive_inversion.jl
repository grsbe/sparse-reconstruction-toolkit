using JLD2
using LinearAlgebra
using Printf
using SparseReconstructionToolkit

include(joinpath(@__DIR__, "..", "src", "gurobi_constrained_lasso_solver.jl"))

const DATA_DIR = joinpath(@__DIR__, "..", "testdata")
const SOLVER_ABSTOL = 1e-7
const SOLVER_RELTOL = 1e-5
const ADMM_MAXITER = 500
const FISTA_MAXITER = 1_000

load_single(name) = JLD2.load(joinpath(DATA_DIR, name))["single_stored_object"]

function timed(f)
    t0 = time_ns()
    result = f()
    return result, (time_ns() - t0) / 1e9
end

function warm_up()
    A = Matrix{Float64}(I, 3, 3)
    b = [1.0, 0.0, 0.0]
    basis_pursuit_denoising_gurobi(A, b; noise=0.1, positive=true)
    nonnegative_constrained_lasso_admm(A, b, 1.0; rho=1.0)
    nonnegative_constrained_lasso_fista(A, b, 1.0)
end

function report(name, A, y, x_true, noise, radius, x, seconds; info=nothing)
    residual = norm(A * x - y)
    reconstruction_error = norm(x - x_true) / max(norm(x_true), eps())
    residual_gap = residual - noise
    radius_gap = norm(x, 1) - radius
    iterations = isnothing(info) ? "-" : string(info.iterations)
    converged = isnothing(info) ? "-" : string(info.converged)
    @printf(
        "%-28s time=%9.3f s  iter=%6s  converged=%5s  rel_x_err=%9.4g  residual=%9.4g  residual-noise=%9.3g  l1=%9.4g  l1-radius=%9.3g  min=%9.3g\n",
        name,
        seconds,
        iterations,
        converged,
        reconstruction_error,
        residual,
        residual_gap,
        norm(x, 1),
        radius_gap,
        minimum(x),
    )
end

function compare_measurement(A, x_true, snr)
    y = load_single("y_noisy_SNR$(snr).jld2")
    noise = load_single("L2_norm_of_noise_SNR$(snr).jld2")

    println("SNR $(snr)")
    println("noise_l2=$(noise), target_residual=$(norm(A * x_true - y))")

    x_gurobi, gurobi_seconds = timed(
        () -> basis_pursuit_denoising_gurobi(A, y; noise, positive=true),
    )
    radius = norm(x_gurobi, 1)
    println("Gurobi BPDN-calibrated l1 radius=$(radius)")

    admm_solver, admm_setup_seconds = timed(() -> LassoADMMSolver(A, y; rho=1.0))
    admm_info, admm_seconds = timed(
        () -> nonnegative_constrained_lasso_admm(
            admm_solver,
            radius;
            abstol=SOLVER_ABSTOL,
            reltol=SOLVER_RELTOL,
            maxiter=ADMM_MAXITER,
            return_info=true,
        ),
    )

    report("positive Gurobi BPDN", A, y, x_true, noise, radius, x_gurobi, gurobi_seconds)
    report(
        "positive ADMM constrained",
        A,
        y,
        x_true,
        noise,
        radius,
        admm_info.x,
        admm_setup_seconds + admm_seconds;
        info=admm_info,
    )
    @printf("  ADMM setup=%0.3f s solve=%0.3f s\n", admm_setup_seconds, admm_seconds)

    step = fista_step_size(A; method=:power)
    fista_solver, fista_setup_seconds = timed(() -> LassoFISTASolver(A, y; step))
    fista_info, fista_seconds = timed(
        () -> nonnegative_constrained_lasso_fista(
            fista_solver,
            radius;
            abstol=SOLVER_ABSTOL,
            reltol=SOLVER_RELTOL,
            maxiter=FISTA_MAXITER,
            return_info=true,
        ),
    )

    report(
        "positive FISTA constrained",
        A,
        y,
        x_true,
        noise,
        radius,
        fista_info.x,
        fista_setup_seconds + fista_seconds;
        info=fista_info,
    )
    @printf("  FISTA step=%0.6g setup=%0.3f s solve=%0.3f s\n", step, fista_setup_seconds, fista_seconds)
    println()
end

function main()
    BLAS.set_num_threads(1)
    A = load_single("A.jld2")
    # A was built for row-major-equivalent flattening of the stored target image.
    x_true = vec(permutedims(load_single("target.jld2")))
    println("Positive inversion comparison")
    println("A_size=$(size(A)), target_length=$(length(x_true)), BLAS_threads=$(BLAS.get_num_threads())")
    println("ADMM/FISTA abstol=$(SOLVER_ABSTOL), reltol=$(SOLVER_RELTOL), ADMM_maxiter=$(ADMM_MAXITER), FISTA_maxiter=$(FISTA_MAXITER)")
    println()
    warm_up()
    compare_measurement(A, x_true, 30)
    compare_measurement(A, x_true, 15)
end

main()
