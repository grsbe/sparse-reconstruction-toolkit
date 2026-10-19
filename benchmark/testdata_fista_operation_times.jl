using CUDA
using JLD2
using LinearAlgebra
using SparseArrays
using SparseReconstructionToolkit
using Statistics

const DATA_DIR = joinpath(@__DIR__, "..", "testdata")
const REPS = parse(Int, get(ENV, "SRT_OP_REPS", "100"))
const ROUNDS = parse(Int, get(ENV, "SRT_OP_ROUNDS", "5"))
const LAMBDA = 2.84e-7

load_single(name) = JLD2.load(joinpath(DATA_DIR, name))["single_stored_object"]

function time_operation(f, gpu; reps=REPS)
    for _ in 1:3
        f()
    end
    gpu && CUDA.synchronize()
    samples = Float64[]
    for _ in 1:ROUNDS
        t0 = time_ns()
        for _ in 1:reps
            f()
        end
        gpu && CUDA.synchronize()
        push!(samples, (time_ns() - t0) / (1e6 * reps))
    end
    return median(samples)
end

function report_operations(label, A, b, step; gpu=false)
    x = similar(b, size(A, 2))
    fill!(x, one(eltype(x)))
    y = copy(x)
    previous = copy(x)
    residual = similar(b)
    gradient = similar(x)
    prox_input = similar(x)
    threshold = step * LAMBDA

    forward = () -> mul!(residual, A, y)
    subtract = () -> (@. residual = residual - b)
    transpose_product = () -> mul!(gradient, adjoint(A), residual)
    proximal = () -> begin
        @. prox_input = y - step * gradient
        @. x = max(prox_input - threshold, zero(eltype(x)))
    end
    momentum = () -> (@. y = x + 0.5 * (x - previous))
    convergence = () -> begin
        @. prox_input = x - previous
        norm(prox_input)
        norm(x)
        norm(previous)
    end

    # Give each operation realistic input values before its isolated timing.
    forward()
    subtract()
    transpose_product()
    gpu && CUDA.synchronize()

    println(label)
    for (name, operation) in (
        ("A*y", forward),
        ("residual -= b", subtract),
        ("A'*residual", transpose_product),
        ("gradient step + threshold", proximal),
        ("momentum update", momentum),
        ("convergence norms", convergence),
    )
        println("  $(rpad(name, 27)) $(round(time_operation(operation, gpu); digits=4)) ms")
    end

    solver = GPUFISTASolver(A, b; step)
    gpu_nonnegative_lasso_fista(solver, LAMBDA; maxiter=1)
    gpu && CUDA.synchronize()
    iteration = () -> gpu_nonnegative_lasso_fista(
        solver, LAMBDA; maxiter=REPS, abstol=0, reltol=0,
        check_every=REPS, return_info=true,
    )
    full = time_operation(iteration, gpu; reps=1) / REPS
    println("  $(rpad("FISTA iteration (check / $REPS)", 27)) $(round(full; digits=4)) ms")
end

function main()
    REPS > 0 && ROUNDS > 0 || error("REPS and ROUNDS must be positive")
    BLAS.set_num_threads(32)
    CUDA.functional() || error("No functional CUDA device")
    CUDA.device!(parse(Int, get(ENV, "SRT_GPU_DEVICE", "1")))
    CUDA.allowscalar(false)

    A = load_single("A.jld2")
    b = load_single("y_noisy_SNR30.jld2")
    step = fista_step_size(A; method=:power)
    A_sparse = sparse(A)
    b_gpu = CuArray(b)
    A_gpu = CuArray(A)
    A_csr = CUDA.CUSPARSE.CuSparseMatrixCSR(A_sparse)
    CUDA.synchronize()

    println("device=$(CUDA.name(CUDA.device())); A=$(size(A)); eltype=$(eltype(A)); nonzeros=$(nnz(A_sparse)); CPU BLAS threads=$(BLAS.get_num_threads())")
    println("Warmed median of $ROUNDS batches, $REPS repetitions per batch; GPU synchronized after each batch.")
    println("One-time setup (warmed median):")
    println("  CPU dense -> CPU sparse    $(round(time_operation(() -> sparse(A), false; reps=1) / 1000; digits=4)) s")
    println("  CPU dense -> GPU dense     $(round(time_operation(() -> CuArray(A), true; reps=1) / 1000; digits=4)) s")
    println("  CPU sparse -> GPU CSR      $(round(time_operation(() -> CUDA.CUSPARSE.CuSparseMatrixCSR(A_sparse), true; reps=1) / 1000; digits=4)) s")
    println("  CPU b -> GPU b             $(round(time_operation(() -> CuArray(b), true; reps=1) / 1000; digits=6)) s")
    report_operations("CPU dense", A, b, step)
    report_operations("GPU dense", A_gpu, b_gpu, step; gpu=true)
    report_operations("GPU sparse CSR", A_csr, b_gpu, step; gpu=true)
end

main()
