using CUDA
using LinearAlgebra
using Random
using SparseArrays
using SparseReconstructionToolkit
using Statistics

CUDA.functional() || error("No functional CUDA device")
CUDA.device!(parse(Int, get(ENV, "SRT_GPU_DEVICE", "0")))
CUDA.allowscalar(false)

rows = parse(Int, get(ENV, "SRT_GPU_ROWS", "3000"))
cols = parse(Int, get(ENV, "SRT_GPU_COLS", "1000"))
precision = get(ENV, "SRT_GPU_PRECISION", "Float32")
T = precision == "Float32" ? Float32 :
    precision == "Float64" ? Float64 :
    error("SRT_GPU_PRECISION must be Float32 or Float64")
zero_fraction = parse(T, get(ENV, "SRT_GPU_ZERO_FRACTION", "0.5"))
iterations = parse(Int, get(ENV, "SRT_GPU_ITERATIONS", "200"))
rows > 0 && cols > 0 && iterations > 0 || error("Dimensions and iterations must be positive")
0 <= zero_fraction < 1 || error("Zero fraction must lie in [0, 1)")
Random.seed!(20261019)

A = randn(T, rows, cols) / sqrt(T(rows))
A[rand(T, rows, cols) .< zero_fraction] .= 0
b = randn(T, rows)
lambda = T(0.01) * norm(A' * b, Inf)
step = fista_step_size(A; method=:power, iterations=100, safety=1.2)
cpu_setup_start = time_ns()
cpu = LassoFISTASolver(A, b; step)
cpu_setup = (time_ns() - cpu_setup_start) / 1e9
matched_cpu = GPUFISTASolver(A, b; step)
dense_setup_start = time_ns()
b_gpu = CuArray(b)
dense_gpu = GPUFISTASolver(CuArray(A), b_gpu; step)
CUDA.synchronize()
dense_setup = (time_ns() - dense_setup_start) / 1e9
sparse_setup_start = time_ns()
sparse_gpu = GPUFISTASolver(CUDA.CUSPARSE.CuSparseMatrixCSR(sparse(A)), b_gpu; step)
CUDA.synchronize()
sparse_setup = (time_ns() - sparse_setup_start) / 1e9

function time_solve(solver, solve; samples=3)
    solve(solver; maxiter=1, abstol=0, reltol=0)
    times = Float64[]
    result = nothing
    for _ in 1:samples
        t0 = time_ns()
        result = solve(solver; maxiter=iterations, abstol=0, reltol=0)
        solver isa GPUFISTASolver && solver.b isa CuArray && CUDA.synchronize()
        push!(times, (time_ns() - t0) / 1e9)
    end
    return result, median(times)
end

cpu_x, cpu_time = time_solve(cpu, (s; kwargs...) -> lasso_fista(s, lambda; kwargs...))
matched_cpu_x, matched_cpu_time = time_solve(matched_cpu, (s; kwargs...) -> gpu_lasso_fista(s, lambda; kwargs...))
dense_x, dense_time = time_solve(dense_gpu, (s; kwargs...) -> gpu_lasso_fista(s, lambda; kwargs...))
sparse_x, sparse_time = time_solve(sparse_gpu, (s; kwargs...) -> gpu_lasso_fista(s, lambda; kwargs...))

println("device=$(CUDA.device()) $(CUDA.name(CUDA.device()))")
println("matrix=$(rows)x$(cols), precision=$(precision), zeros=$(round(count(iszero, A) / length(A), digits=3)), iterations=$(iterations), BLAS threads=$(BLAS.get_num_threads())")
println("CPU existing setup=$(round(cpu_setup, digits=4)) s; solve=$(round(cpu_time, digits=4)) s")
println("CPU same-loop solve=$(round(matched_cpu_time, digits=4)) s")
println("GPU dense setup=$(round(dense_setup, digits=4)) s; solve=$(round(dense_time, digits=4)) s; speedup vs existing CPU=$(round(cpu_time/dense_time, digits=2))x, vs same-loop CPU=$(round(matched_cpu_time/dense_time, digits=2))x")
println("GPU CSR setup=$(round(sparse_setup, digits=4)) s; solve=$(round(sparse_time, digits=4)) s; speedup vs existing CPU=$(round(cpu_time/sparse_time, digits=2))x, vs same-loop CPU=$(round(matched_cpu_time/sparse_time, digits=2))x")
println("dense max coefficient error=$(maximum(abs.(Array(dense_x) - cpu_x)))")
println("same-loop CPU max coefficient error=$(maximum(abs.(matched_cpu_x - cpu_x)))")
println("CSR max coefficient error=$(maximum(abs.(Array(sparse_x) - cpu_x)))")
