using LinearAlgebra
using Test
using srt

@testset "penalized LASSO ADMM" begin
    A = Matrix{Float64}(I, 3, 3)
    b = [2.0, -1.0, 0.2]
    info = lasso_admm(
        A,
        b,
        0.5;
        rho=1.0,
        abstol=1e-9,
        reltol=1e-8,
        return_info=true,
    )

    workspace = LassoADMMWorkspace(A, b; rho=1.0)
    solver = LassoADMMSolver(A, b; rho=1.0)
    workspace_info = lasso_admm!(
        workspace,
        0.5;
        abstol=1e-9,
        reltol=1e-8,
        return_info=true,
    )
    warm_info = lasso_admm!(workspace, 0.5; warm_start=true, return_info=true)
    solver_info = lasso_admm(
        solver,
        0.5;
        abstol=1e-9,
        reltol=1e-8,
        return_info=true,
    )
    solver_warm_info = lasso_admm(solver, 0.5; warm_start=true, return_info=true)

    @test info.converged
    @test info.x ≈ [1.5, -0.5, 0.0] atol = 2e-6
    @test workspace_info.x ≈ info.x atol = 2e-6
    @test solver_info.x ≈ info.x atol = 2e-6
    @test warm_info.iterations <= workspace_info.iterations
    @test solver_warm_info.iterations <= solver_info.iterations
    @test lasso_admm(A, b, 0.5; rho=1.0) isa Vector
end

@testset "constrained LASSO ADMM" begin
    A = Matrix{Float64}(I, 3, 3)
    b = [2.0, -1.0, 0.0]
    info = constrained_lasso_admm(
        A,
        b,
        1.0;
        rho=1.0,
        abstol=1e-9,
        reltol=1e-8,
        return_info=true,
    )
    workspace = LassoADMMWorkspace(A, b; rho=1.0)
    solver = LassoADMMSolver(A, b; rho=1.0)
    workspace_x = constrained_lasso_admm!(workspace, 1.0)
    solver_x = constrained_lasso_admm(solver, 1.0)

    @test info.converged
    @test info.x ≈ [1.0, 0.0, 0.0] atol = 2e-6
    @test workspace_x ≈ info.x atol = 2e-4
    @test solver_x ≈ info.x atol = 2e-4
    @test norm(info.x, 1) <= 1.0 + 2e-6
end

@testset "nonnegative penalized LASSO ADMM" begin
    A = Matrix{Float64}(I, 3, 3)
    b = [2.0, -1.0, 0.2]
    info = nonnegative_lasso_admm(
        A,
        b,
        0.5;
        rho=1.0,
        abstol=1e-9,
        reltol=1e-8,
        return_info=true,
    )
    workspace = LassoADMMWorkspace(A, b; rho=1.0)
    solver = LassoADMMSolver(A, b; rho=1.0)
    workspace_x = nonnegative_lasso_admm!(workspace, 0.5)
    solver_x = nonnegative_lasso_admm(solver, 0.5)

    @test info.converged
    @test info.x ≈ [1.5, 0.0, 0.0] atol = 2e-6
    @test workspace_x ≈ info.x atol = 2e-4
    @test solver_x ≈ info.x atol = 2e-4
    @test all(info.x .>= -2e-6)
end

@testset "nonnegative constrained LASSO ADMM" begin
    A = Matrix{Float64}(I, 3, 3)
    b = [2.0, -1.0, 0.0]
    info = nonnegative_constrained_lasso_admm(
        A,
        b,
        1.0;
        rho=1.0,
        abstol=1e-9,
        reltol=1e-8,
        return_info=true,
    )
    workspace = LassoADMMWorkspace(A, b; rho=1.0)
    solver = LassoADMMSolver(A, b; rho=1.0)
    workspace_x = nonnegative_constrained_lasso_admm!(workspace, 1.0)
    solver_x = nonnegative_constrained_lasso_admm(solver, 1.0)

    @test info.converged
    @test info.x ≈ [1.0, 0.0, 0.0] atol = 2e-6
    @test workspace_x ≈ info.x atol = 2e-4
    @test solver_x ≈ info.x atol = 2e-4
    @test all(info.x .>= -2e-6)
    @test norm(info.x, 1) <= 1.0 + 2e-6
end

@testset "penalized LASSO FISTA" begin
    A = Matrix{Float64}(I, 3, 3)
    b = [2.0, -1.0, 0.2]
    info = lasso_fista(
        A,
        b,
        0.5;
        abstol=1e-9,
        reltol=1e-8,
        return_info=true,
    )
    workspace = LassoFISTAWorkspace(A, b)
    solver = LassoFISTASolver(A, b)
    power_step = fista_step_size(A; method=:power)
    power_x = lasso_fista(
        A,
        b,
        0.5;
        step=power_step,
        abstol=1e-9,
        reltol=1e-8,
    )
    workspace_x = lasso_fista!(workspace, 0.5)
    solver_x = lasso_fista(solver, 0.5)
    admm_x = lasso_admm(A, b, 0.5; rho=1.0)

    @test info.converged
    @test info.x ≈ [1.5, -0.5, 0.0] atol = 2e-6
    @test workspace_x ≈ info.x atol = 2e-6
    @test solver_x ≈ info.x atol = 2e-6
    @test power_x ≈ info.x atol = 2e-6
    @test fista_step_size(A) ≈ 1.0
    @test 0 < power_step < fista_step_size(A)
    @test info.x ≈ admm_x atol = 2e-4
end

@testset "constrained LASSO FISTA" begin
    A = Matrix{Float64}(I, 3, 3)
    b = [2.0, -1.0, 0.0]
    info = constrained_lasso_fista(
        A,
        b,
        1.0;
        abstol=1e-9,
        reltol=1e-8,
        return_info=true,
    )
    solver = LassoFISTASolver(A, b)
    solver_x = constrained_lasso_fista(solver, 1.0)

    @test info.converged
    @test info.x ≈ [1.0, 0.0, 0.0] atol = 2e-6
    @test solver_x ≈ info.x atol = 2e-6
    @test norm(info.x, 1) <= 1.0 + 2e-6
end

@testset "nonnegative constrained LASSO FISTA" begin
    A = Matrix{Float64}(I, 3, 3)
    b = [2.0, -1.0, 0.0]
    info = nonnegative_constrained_lasso_fista(
        A,
        b,
        1.0;
        abstol=1e-9,
        reltol=1e-8,
        return_info=true,
    )
    solver = LassoFISTASolver(A, b)
    solver_x = nonnegative_constrained_lasso_fista(solver, 1.0)

    @test info.converged
    @test info.x ≈ [1.0, 0.0, 0.0] atol = 2e-6
    @test solver_x ≈ info.x atol = 2e-6
    @test all(info.x .>= -2e-6)
    @test norm(info.x, 1) <= 1.0 + 2e-6
end

@testset "nonnegative penalized LASSO FISTA" begin
    A = Matrix{Float64}(I, 3, 3)
    b = [2.0, -1.0, 0.2]
    info = nonnegative_lasso_fista(
        A,
        b,
        0.5;
        abstol=1e-9,
        reltol=1e-8,
        return_info=true,
    )
    workspace = LassoFISTAWorkspace(A, b)
    solver = LassoFISTASolver(A, b)
    workspace_x = nonnegative_lasso_fista!(workspace, 0.5)
    solver_x = nonnegative_lasso_fista(solver, 0.5)
    admm_x = nonnegative_lasso_admm(A, b, 0.5; rho=1.0)

    @test info.converged
    @test info.x ≈ [1.5, 0.0, 0.0] atol = 2e-6
    @test workspace_x ≈ info.x atol = 2e-6
    @test solver_x ≈ info.x atol = 2e-6
    @test info.x ≈ admm_x atol = 2e-4
    @test all(info.x .>= -2e-6)
end

@testset "ADMM input validation" begin
    A = Matrix{Float64}(I, 2, 2)
    b = ones(2)
    workspace = LassoADMMWorkspace(A, b)
    solver = LassoADMMSolver(A, b)
    fista_workspace = LassoFISTAWorkspace(A, b)
    fista_solver = LassoFISTASolver(A, b)

    @test_throws ArgumentError lasso_admm(A, b, -1.0)
    @test_throws ArgumentError lasso_admm!(workspace, -1.0)
    @test_throws ArgumentError lasso_admm(solver, -1.0)
    @test_throws ArgumentError constrained_lasso_admm(A, b, 0.0)
    @test_throws ArgumentError constrained_lasso_admm!(workspace, 0.0)
    @test_throws ArgumentError nonnegative_lasso_admm(A, b, -1.0)
    @test_throws ArgumentError nonnegative_lasso_admm!(workspace, -1.0)
    @test_throws ArgumentError nonnegative_constrained_lasso_admm(A, b, 0.0)
    @test_throws ArgumentError nonnegative_constrained_lasso_admm!(workspace, 0.0)
    @test_throws ArgumentError lasso_fista(A, b, -1.0)
    @test_throws ArgumentError lasso_fista!(fista_workspace, -1.0)
    @test_throws ArgumentError lasso_fista(fista_solver, -1.0)
    @test_throws ArgumentError nonnegative_lasso_fista(A, b, -1.0)
    @test_throws ArgumentError nonnegative_lasso_fista!(fista_workspace, -1.0)
    @test_throws ArgumentError constrained_lasso_fista(A, b, 0.0)
    @test_throws ArgumentError constrained_lasso_fista!(fista_workspace, 0.0)
    @test_throws ArgumentError nonnegative_constrained_lasso_fista(A, b, 0.0)
    @test_throws ArgumentError nonnegative_constrained_lasso_fista!(fista_workspace, 0.0)
    @test_throws ArgumentError LassoFISTAWorkspace(A, b; step=0.0)
    @test_throws ArgumentError fista_step_size(A; method=:unknown)
    @test_throws ArgumentError fista_step_size(A; method=:power, iterations=0)
    @test_throws ArgumentError fista_step_size(A; method=:power, safety=0.0)
    @test_throws ArgumentError lasso_admm(A, b, 1.0; rho=0.0)
    @test_throws DimensionMismatch lasso_admm(A, ones(3), 1.0)
    @test_throws DimensionMismatch lasso_admm(A, b, 1.0; x0=zeros(3))
end
