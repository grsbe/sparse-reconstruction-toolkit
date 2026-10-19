@testset "device-ready FISTA parity on CPU arrays" begin
    A = [1.0 0.0 0.2; 0.0 1.0 0.1; 0.3 0.0 1.0; 0.0 0.4 0.0]
    b = [2.0, -1.0, 0.5, 0.1]
    step = fista_step_size(A)
    for (device_solve, cpu_solve) in (
        (gpu_lasso_fista, lasso_fista),
        (gpu_nonnegative_lasso_fista, nonnegative_lasso_fista),
    )
        reference = cpu_solve(A, b, 0.15; step, abstol=1e-10,
                              reltol=1e-9, maxiter=10_000)
        result = device_solve(A, b, 0.15; step, abstol=1e-10,
                              reltol=1e-9, maxiter=10_000,
                              return_info=true)
        @test result.converged
        @test result.x ≈ reference atol=1e-8
        solver = GPUFISTASolver(A, b; step)
        first = device_solve(solver, 0.15; abstol=1e-10,
                             reltol=1e-9, maxiter=10_000, return_info=true)
        @test first.converged
        @test first.x ≈ reference atol=1e-8
        warm = device_solve(solver, 0.1; warm_start=true, abstol=1e-10,
                            reltol=1e-9, maxiter=10_000, return_info=true)
        @test warm.converged
        @test warm.x ≈ cpu_solve(A, b, 0.1; step, abstol=1e-10,
                                reltol=1e-9, maxiter=10_000) atol=1e-8
    end
    @test GPUFISTASolver(A, b).step > 0
    @test_throws ArgumentError GPUFISTASolver(A, b; step=0)
    @test_throws ArgumentError GPUFISTASolver(Float32.(A), b)
    @test_throws DimensionMismatch GPUFISTASolver(A, b; x0=zeros(2))
end
