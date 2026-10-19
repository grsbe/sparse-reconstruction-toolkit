using CUDA
using JLD2
using LinearAlgebra
using SparseArrays
using SparseReconstructionToolkit
using Statistics

const DATA_DIR = joinpath(@__DIR__, "..", "testdata")
const LAMBDA = 2.84e-7
const ABSTOL = 1e-7
const RELTOL = 1e-5
const MAXITER = parse(Int, get(ENV, "SRT_GPU_MAXITER", "10000"))
const SAMPLES = parse(Int, get(ENV, "SRT_GPU_SAMPLES", "3"))

load_single(name) = JLD2.load(joinpath(DATA_DIR, name))["single_stored_object"]

function timed_solve(solver, solve)
    solve(solver; maxiter=1, abstol=ABSTOL, reltol=RELTOL,
          return_info=true)
    times = Float64[]
    info = nothing
    for _ in 1:SAMPLES
        t0 = time_ns()
        info = solve(solver; maxiter=MAXITER, abstol=ABSTOL,
                     reltol=RELTOL, return_info=true)
        solver isa GPUFISTASolver && solver.b isa CuArray && CUDA.synchronize()
        push!(times, (time_ns() - t0) / 1e9)
    end
    return info, median(times), times
end

function main()
    MAXITER > 0 && SAMPLES > 0 || error("MAXITER and SAMPLES must be positive")
    BLAS.set_num_threads(32)
    CUDA.functional() || error("No functional CUDA device")
    CUDA.device!(parse(Int, get(ENV, "SRT_GPU_DEVICE", "0")))
    CUDA.allowscalar(false)

    A = load_single("A.jld2")
    y = load_single("y_noisy_SNR30.jld2")
    x_true = vec(permutedims(load_single("target.jld2")))
    noise = load_single("L2_norm_of_noise_SNR30.jld2")
    step = fista_step_size(A; method=:power)

    t0 = time_ns()
    cpu = LassoFISTASolver(A, y; step)
    cpu_setup = (time_ns() - t0) / 1e9
    cpu_same = GPUFISTASolver(A, y; step)

    t0 = time_ns()
    y_gpu = CuArray(y)
    dense = GPUFISTASolver(CuArray(A), y_gpu; step)
    CUDA.synchronize()
    dense_setup = (time_ns() - t0) / 1e9

    t0 = time_ns()
    csr = GPUFISTASolver(CUDA.CUSPARSE.CuSparseMatrixCSR(sparse(A)),
                         y_gpu; step)
    CUDA.synchronize()
    csr_setup = (time_ns() - t0) / 1e9

    cpu_info, cpu_seconds, cpu_samples =
        timed_solve(cpu, (s; kwargs...) ->
            nonnegative_lasso_fista(s, LAMBDA; kwargs...))
    same_info, same_seconds, same_samples =
        timed_solve(cpu_same, (s; kwargs...) ->
            gpu_nonnegative_lasso_fista(s, LAMBDA; kwargs...))
    dense_info, dense_seconds, dense_samples =
        timed_solve(dense, (s; kwargs...) ->
            gpu_nonnegative_lasso_fista(s, LAMBDA; kwargs...))
    csr_info, csr_seconds, csr_samples =
        timed_solve(csr, (s; kwargs...) ->
            gpu_nonnegative_lasso_fista(s, LAMBDA; kwargs...))

    println("device=$(CUDA.device()) $(CUDA.name(CUDA.device()))")
    println("A=$(size(A)), eltype=$(eltype(A)), zero_fraction=$(count(iszero, A)/length(A)), BLAS_threads=$(BLAS.get_num_threads())")
    println("SNR=30 lambda=$(LAMBDA), step=$(step), noise=$(noise)")
    println("CPU existing: setup=$(cpu_setup) s, solve_median=$(cpu_seconds) s, samples=$(cpu_samples), iterations=$(cpu_info.iterations), converged=$(cpu_info.converged)")
    println("CPU same-loop: solve_median=$(same_seconds) s, samples=$(same_samples), iterations=$(same_info.iterations), converged=$(same_info.converged)")
    println("GPU dense: setup=$(dense_setup) s, solve_median=$(dense_seconds) s, samples=$(dense_samples), iterations=$(dense_info.iterations), converged=$(dense_info.converged)")
    println("GPU CSR: setup=$(csr_setup) s, solve_median=$(csr_seconds) s, samples=$(csr_samples), iterations=$(csr_info.iterations), converged=$(csr_info.converged)")

    x_cpu = cpu_info.x
    x_matched = same_info.x
    for (label, x) in (("CPU same-loop", same_info.x),
                       ("GPU dense", Array(dense_info.x)),
                       ("GPU CSR", Array(csr_info.x)))
        println("$label: rel_to_cpu=$(norm(x-x_cpu)/max(norm(x_cpu),eps())), rel_to_same_loop=$(norm(x-x_matched)/max(norm(x_matched),eps())), rel_to_target=$(norm(x-x_true)/max(norm(x_true),eps())), residual=$(norm(A*x-y))")
    end
end

main()
