using JLD2
using LinearAlgebra
using Printf
using Random
using Statistics
using SparseReconstructionToolkit

const DATA_DIR = joinpath(@__DIR__, "..", "testdata")
const BLAS_THREADS = 32
const FISTA_SAMPLES = parse(Int, get(ENV, "SRT_FISTA_SAMPLES", "3"))
const FISTA_MAXITER = parse(Int, get(ENV, "SRT_FISTA_MAXITER", "2000"))
const SOLVER_ABSTOL = 1e-7
const SOLVER_RELTOL = 1e-5
const SPARSE_NONZEROS = 40
const SMOOTH_NOISE_LEVEL = 0.01
const L1_FRACTION = 5e-3
const RIDGE_FRACTION = 1e-3

load_single(name) = JLD2.load(joinpath(DATA_DIR, name))["single_stored_object"]

function timed(f)
    t0 = time_ns()
    result = f()
    return result, (time_ns() - t0) / 1e9
end

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

function gaussian_column(rows, cols, center_row, center_col, width_row, width_col)
    image = Matrix{Float64}(undef, rows, cols)
    for col in 1:cols, row in 1:rows
        row_distance = (row - center_row) / width_row
        col_distance = (col - center_col) / width_col
        image[row, col] = exp(-0.5 * (row_distance^2 + col_distance^2))
    end
    image ./= maximum(image)
    return vec(permutedims(image))
end

function smooth_basis(rows, cols)
    centers = (
        (0.18, 0.22, 0.17, 0.20),
        (0.22, 0.72, 0.20, 0.16),
        (0.48, 0.46, 0.24, 0.25),
        (0.62, 0.82, 0.18, 0.17),
        (0.78, 0.24, 0.16, 0.22),
        (0.86, 0.62, 0.14, 0.19),
    )
    S = Matrix{Float64}(undef, rows * cols, length(centers))
    for (index, (row, col, row_width, col_width)) in enumerate(centers)
        S[:, index] = gaussian_column(
            rows,
            cols,
            1 + row * (rows - 1),
            1 + col * (cols - 1),
            row_width * rows,
            col_width * cols,
        )
    end
    return S
end

function toy_problem(A, rows, cols; seed=20260522)
    rng = MersenneTwister(seed)
    S = smooth_basis(rows, cols)
    B, B_seconds = timed(() -> A * S)
    x_true = zeros(size(A, 2))
    support = randperm(rng, length(x_true))[1:SPARSE_NONZEROS]
    x_true[support] .= 100.0 .+ 150.0 * rand(rng, SPARSE_NONZEROS)
    z_true = 0.25 .+ rand(rng, size(S, 2))
    sparse_signal = A * x_true
    smooth_signal = B * z_true
    clean = sparse_signal + smooth_signal
    noise_sigma = SMOOTH_NOISE_LEVEL * norm(clean) / sqrt(length(clean))
    y = clean + noise_sigma * randn(rng, length(clean))
    return B, B_seconds, y, x_true, z_true, clean, sparse_signal, smooth_signal
end

positive_lambda_max(A, y) = max(maximum(transpose(A) * y), eps())

function ridge_curvature(B)
    return maximum(sum(abs2, B; dims=1))
end

function objective(A, B, y, lambda_x, lambda_z, x, z)
    return 0.5 * norm(A * x + B * z - y)^2 +
           lambda_x * norm(x, 1) + 0.5 * lambda_z * norm(z)^2
end

relative_error(value, truth) = norm(value - truth) / max(norm(truth), eps())

function warm_up()
    A = Matrix{Float64}(I, 4, 4)
    B = A[:, 1:2]
    y = ones(4)
    lasso_ridge_fista(
        A,
        B,
        y,
        0.1,
        0.1;
        x_nonnegative=true,
        z_nonnegative=true,
        maxiter=5,
    )
end

function main()
    BLAS.set_num_threads(BLAS_THREADS)
    A = load_single("A.jld2")
    target = load_single("target.jld2")
    B, B_seconds, y, x_true, z_true, clean, sparse_signal, smooth_signal =
        toy_problem(A, size(target)...)
    lambda_x = L1_FRACTION * positive_lambda_max(A, y)
    lambda_z = RIDGE_FRACTION * ridge_curvature(B)
    warm_up()
    step, step_seconds = timed(() -> fista_step_size(A, B; method=:power))

    println("Sparse plus smooth nonnegative LASSO-ridge FISTA toy benchmark")
    println("A_size=$(size(A)), B_size=$(size(B)), BLAS_threads=$(BLAS.get_num_threads())")
    @printf(
        "clean_norm=%9.4g sparse_norm=%9.4g smooth_norm=%9.4g\n",
        norm(clean),
        norm(sparse_signal),
        norm(smooth_signal),
    )
    @printf(
        "lambda_x=%9.3g lambda_z=%9.3g power_step=%9.3g\n",
        lambda_x,
        lambda_z,
        step,
    )
    @printf("smooth B build=%0.3f s, power step=%0.3f s\n", B_seconds, step_seconds)
    println("maxiter=$(FISTA_MAXITER), abstol=$(SOLVER_ABSTOL), reltol=$(SOLVER_RELTOL)")
    println()

    solver, solver_seconds = timed(() -> LassoRidgeFISTASolver(A, B, y; step))
    info, solve_seconds, solve_samples = timed_samples(
        () -> lasso_ridge_fista(
            solver,
            lambda_x,
            lambda_z;
            x_nonnegative=true,
            z_nonnegative=true,
            abstol=SOLVER_ABSTOL,
            reltol=SOLVER_RELTOL,
            maxiter=FISTA_MAXITER,
            return_info=true,
        );
        samples=FISTA_SAMPLES,
    )

    @printf("solver setup=%0.3f s\n", solver_seconds)
    @printf("solve median=%0.3f s samples=%s\n", solve_seconds, string(solve_samples))
    @printf(
        "termination: iterations=%d converged=%s iterate_change=%9.3g\n",
        info.iterations,
        string(info.converged),
        info.iterate_change,
    )
    @printf(
        "objective=%9.5g residual=%9.4g rel_x_err=%9.4g rel_z_err=%9.4g x_nnz=%d min_x=%9.3g min_z=%9.3g\n",
        objective(A, B, y, lambda_x, lambda_z, info.x, info.z),
        norm(A * info.x + B * info.z - y),
        relative_error(info.x, x_true),
        relative_error(info.z, z_true),
        count(>(1e-8), info.x),
        minimum(info.x),
        minimum(info.z),
    )
end

main()
