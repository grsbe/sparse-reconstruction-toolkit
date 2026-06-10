using JLD2
using LinearAlgebra
using Printf
using SparseReconstructionToolkit

const DATA_DIR = joinpath(@__DIR__, "..", "testdata_robustness")
const SNR = parse(Int, get(ENV, "SRT_ROBUST_SNR", "15"))
const LAMBDA = parse(Float64, get(ENV, "SRT_ROBUST_LAMBDA", "1e-6"))
const TAU = parse(Float64, get(ENV, "SRT_ROBUST_TAU", "1.0"))
const MAXITER = parse(Int, get(ENV, "SRT_ROBUST_MAXITER", "20"))
const ABSTOL = parse(Float64, get(ENV, "SRT_ROBUST_ABSTOL", "1e-7"))
const RELTOL = parse(Float64, get(ENV, "SRT_ROBUST_RELTOL", "1e-5"))
const THREAD_COUNTS = parse.(Int, split(get(ENV, "SRT_ROBUST_THREADS", "1,4,16,32"), ","))

load_single(name) = JLD2.load(joinpath(DATA_DIR, name))["single_stored_object"]
imagevec(x) = vec(permutedims(x))

function timed(f)
    t0 = time_ns()
    result = f()
    return result, (time_ns() - t0) / 1e9
end

function relative_error(x, x_true)
    return norm(x - x_true) / max(norm(x_true), eps())
end

function recovery_summary(A_base, y, x, x_true)
    residual = norm(A_base * x - y)
    target_residual = norm(A_base * x_true - y)
    return (
        residual=residual,
        target_residual=target_residual,
        residual_ratio=residual / max(target_residual, eps()),
        rel_x_error=relative_error(x, x_true),
        l1=norm(x, 1),
        nonzeros=count(abs.(x) .> 1e-8),
        support100=count(abs.(x) .> 0.01 * maximum(abs.(x_true))),
    )
end

function print_case(name, setup_seconds, solve_seconds, info, summary)
    @printf(
        "%-24s setup=%7.3f s  solve=%7.3f s  per_iter=%8.5f s  iter=%5d  conv=%5s  backtracks=%4d  step=%9.3g\n",
        name,
        setup_seconds,
        solve_seconds,
        solve_seconds / max(info.iterations, 1),
        info.iterations,
        string(info.converged),
        info.backtracks,
        info.step,
    )
    @printf(
        "%-24s residual=%9.4g  target_residual=%9.4g  residual/target=%8.4g  rel_x_err=%8.4g  l1=%9.4g  nnz=%5d  support01=%5d\n",
        "",
        summary.residual,
        summary.target_residual,
        summary.residual_ratio,
        summary.rel_x_error,
        summary.l1,
        summary.nonzeros,
        summary.support100,
    )
end

function run_case(A_rot, A_base, y, x_true; copy_slices)
    workspace, setup_seconds = timed(() -> SoftminLassoWorkspace(A_rot, y; copy_slices))

    # Compile the hot path outside the measured solve. Reset to zero afterward so
    # each reported run starts from the same initial iterate.
    softmin_lasso!(
        workspace,
        LAMBDA;
        tau=TAU,
        maxiter=1,
        return_info=true,
        warm_start=false,
    )
    fill!(workspace.x, 0.0)

    info, solve_seconds = timed(
        () -> softmin_lasso!(
            workspace,
            LAMBDA;
            tau=TAU,
            maxiter=MAXITER,
            abstol=ABSTOL,
            reltol=RELTOL,
            return_info=true,
            warm_start=false,
        ),
    )
    summary = recovery_summary(A_base, y, info.x, x_true)
    return setup_seconds, solve_seconds, info, summary
end

function main()
    A_base = load_single("A.jld2")
    A_rot = load_single("A_rot_multimatrix.jld2")
    prior = load_single("prior.jld2")
    target = load_single("target.jld2")
    y_noisy = load_single("y_noisy$(SNR).jld2")
    noise = load_single("L2noise$(SNR).jld2")
    noise_norm = noise isa Number ? float(noise) : norm(noise)

    x_prior = imagevec(prior)
    x_true = imagevec(target)
    y = y_noisy - A_base * x_prior

    println("Robust soft-min LASSO benchmark on testdata_robustness")
    println("A_base_size=$(size(A_base)), A_rot_size=$(size(A_rot)), target_length=$(length(x_true))")
    println("SNR=$(SNR), lambda=$(LAMBDA), tau=$(TAU), maxiter=$(MAXITER), abstol=$(ABSTOL), reltol=$(RELTOL)")
    println("noise_l2=$(noise_norm), residualized_target_residual=$(norm(A_base * x_true - y))")
    println("thread_counts=$(THREAD_COUNTS)")
    println()

    for threads in THREAD_COUNTS
        BLAS.set_num_threads(threads)
        println("BLAS threads=$(BLAS.get_num_threads())")
        for copy_slices in (true, false)
            setup_seconds, solve_seconds, info, summary = run_case(
                A_rot,
                A_base,
                y,
                x_true;
                copy_slices,
            )
            name = copy_slices ? "copied slices" : "view slices"
            print_case(name, setup_seconds, solve_seconds, info, summary)
        end
        println()
    end
end

main()
