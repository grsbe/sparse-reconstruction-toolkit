using JLD2
using LinearAlgebra
using Printf
using SparseReconstructionToolkit

const DATA_DIR = joinpath(@__DIR__, "..", "testdata")
const BLAS_THREADS = 32
const SEARCH_STEPS = 10
const FISTA_MAXITER = 2_000
const ADMM_MAXITER = 500
const SOLVER_ABSTOL = 1e-7
const SOLVER_RELTOL = 1e-5

load_single(name) = JLD2.load(joinpath(DATA_DIR, name))["single_stored_object"]

function timed(f)
    t0 = time_ns()
    result = f()
    return result, (time_ns() - t0) / 1e9
end

positive_lambda_max(A, y) = max(maximum(transpose(A) * y), eps())
relative_x_error(x, x_true) = norm(x - x_true) / max(norm(x_true), eps())

function residual(A, y, x)
    return norm(A * x - y)
end

function better_candidate(A, y, noise, best, lambda, info, seconds)
    residual_value = residual(A, y, info.x)
    candidate = (
        lambda=lambda,
        info=info,
        seconds=seconds,
        residual=residual_value,
        gap=residual_value - noise,
    )
    if isnothing(best) || abs(candidate.gap) < abs(best.gap)
        return candidate
    end
    return best
end

function search_positive_fista(A, y, noise)
    step = fista_step_size(A; method=:power)
    solver = LassoFISTASolver(A, y; step)
    lambda_low = 0.0
    lambda_high = positive_lambda_max(A, y)
    best = nothing
    total_seconds = 0.0
    println("FISTA lambda search: step=$(step), lambda_high=$(lambda_high)")

    for trial in 1:SEARCH_STEPS
        lambda = trial == 1 ? lambda_high / 2 : (lambda_low + lambda_high) / 2
        info, seconds = timed(
            () -> nonnegative_lasso_fista(
                solver,
                lambda;
                warm_start=trial > 1,
                abstol=SOLVER_ABSTOL,
                reltol=SOLVER_RELTOL,
                maxiter=FISTA_MAXITER,
                return_info=true,
            ),
        )
        total_seconds += seconds
        best = better_candidate(A, y, noise, best, lambda, info, seconds)
        gap = residual(A, y, info.x) - noise
        @printf(
            "  trial=%2d lambda=%9.3g residual=%9.4g gap=%9.3g iter=%5d converged=%5s time=%7.3f s\n",
            trial,
            lambda,
            gap + noise,
            gap,
            info.iterations,
            string(info.converged),
            seconds,
        )
        if gap > 0
            lambda_high = lambda
        else
            lambda_low = lambda
        end
    end

    return best, total_seconds
end

function search_positive_admm(A, y, noise; rho=0.01)
    solver = LassoADMMSolver(A, y; rho)
    lambda_low = 0.0
    lambda_high = positive_lambda_max(A, y)
    best = nothing
    total_seconds = 0.0
    println("ADMM lambda search: rho=$(rho), lambda_high=$(lambda_high)")

    for trial in 1:SEARCH_STEPS
        lambda = trial == 1 ? lambda_high / 2 : (lambda_low + lambda_high) / 2
        info, seconds = timed(
            () -> nonnegative_lasso_admm(
                solver,
                lambda;
                warm_start=trial > 1,
                abstol=SOLVER_ABSTOL,
                reltol=SOLVER_RELTOL,
                maxiter=ADMM_MAXITER,
                return_info=true,
            ),
        )
        total_seconds += seconds
        best = better_candidate(A, y, noise, best, lambda, info, seconds)
        gap = residual(A, y, info.x) - noise
        @printf(
            "  trial=%2d lambda=%9.3g residual=%9.4g gap=%9.3g iter=%5d converged=%5s time=%7.3f s\n",
            trial,
            lambda,
            gap + noise,
            gap,
            info.iterations,
            string(info.converged),
            seconds,
        )
        if gap > 0
            lambda_high = lambda
        else
            lambda_low = lambda
        end
    end

    return best, total_seconds
end

function report_best(name, A, y, x_true, noise, best, total_seconds)
    @printf(
        "%-22s lambda=%9.3g residual=%9.4g residual-noise=%9.3g rel_x_err=%9.4g l1=%9.4g min=%9.3g best_solve=%7.3f s search_total=%7.3f s iter=%5d converged=%5s\n",
        name,
        best.lambda,
        best.residual,
        best.gap,
        relative_x_error(best.info.x, x_true),
        norm(best.info.x, 1),
        minimum(best.info.x),
        best.seconds,
        total_seconds,
        best.info.iterations,
        string(best.info.converged),
    )
end

function compare_measurement(A, x_true, snr)
    y = load_single("y_noisy_SNR$(snr).jld2")
    noise = load_single("L2_norm_of_noise_SNR$(snr).jld2")
    println("SNR $(snr): noise=$(noise), target_residual=$(residual(A, y, x_true))")
    fista_best, fista_total = search_positive_fista(A, y, noise)
    admm_best, admm_total = search_positive_admm(A, y, noise)
    report_best("positive FISTA", A, y, x_true, noise, fista_best, fista_total)
    report_best("positive ADMM", A, y, x_true, noise, admm_best, admm_total)
    println()
end

function main()
    BLAS.set_num_threads(BLAS_THREADS)
    A = load_single("A.jld2")
    x_true = vec(permutedims(load_single("target.jld2")))
    println("Positive penalized LASSO noise matching")
    println("A_size=$(size(A)), BLAS_threads=$(BLAS.get_num_threads()), search_steps=$(SEARCH_STEPS)")
    println("FISTA_maxiter=$(FISTA_MAXITER), ADMM_maxiter=$(ADMM_MAXITER), abstol=$(SOLVER_ABSTOL), reltol=$(SOLVER_RELTOL)")
    println()
    compare_measurement(A, x_true, 30)
    compare_measurement(A, x_true, 15)
end

main()
