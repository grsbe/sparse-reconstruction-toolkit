using LinearAlgebra
using Test
using SparseReconstructionToolkit

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

@testset "LASSO-ridge FISTA" begin
    A = [Matrix{Float64}(I, 2, 2); zeros(2, 2)]
    B = [zeros(2, 2); Matrix{Float64}(I, 2, 2)]
    b = [2.0, -1.0, 3.0, -4.0]
    info = lasso_ridge_fista(
        A,
        B,
        b,
        0.5,
        1.0;
        abstol=1e-9,
        reltol=1e-8,
        return_info=true,
    )
    positive_info = lasso_ridge_fista(
        A,
        B,
        b,
        0.5,
        1.0;
        x_nonnegative=true,
        z_nonnegative=true,
        abstol=1e-9,
        reltol=1e-8,
        return_info=true,
    )
    named_positive_info = nonnegative_lasso_ridge_fista(
        A,
        B,
        b,
        0.5,
        1.0;
        abstol=1e-9,
        reltol=1e-8,
        return_info=true,
    )
    workspace = LassoRidgeFISTAWorkspace(A, B, b)
    named_workspace = LassoRidgeFISTAWorkspace(A, B, b)
    solver = LassoRidgeFISTASolver(A, B, b)
    named_solver = LassoRidgeFISTASolver(A, B, b)
    workspace_result = lasso_ridge_fista!(workspace, 0.5, 1.0)
    named_workspace_result = nonnegative_lasso_ridge_fista!(named_workspace, 0.5, 1.0)
    warm_info = lasso_ridge_fista!(
        workspace,
        0.5,
        1.0;
        warm_start=true,
        return_info=true,
    )
    solver_result = lasso_ridge_fista(solver, 0.5, 1.0)
    named_solver_result = nonnegative_lasso_ridge_fista(named_solver, 0.5, 1.0)
    power_step = fista_step_size(A, B; method=:power)
    power_result = lasso_ridge_fista(
        A,
        B,
        b,
        0.5,
        1.0;
        step=power_step,
        abstol=1e-9,
        reltol=1e-8,
    )

    @test info.converged
    @test info.x ≈ [1.5, -0.5] atol = 2e-6
    @test info.z ≈ [1.5, -2.0] atol = 2e-6
    @test positive_info.x ≈ [1.5, 0.0] atol = 2e-6
    @test positive_info.z ≈ [1.5, 0.0] atol = 2e-6
    @test named_positive_info.x ≈ positive_info.x atol = 2e-6
    @test named_positive_info.z ≈ positive_info.z atol = 2e-6
    @test all(positive_info.x .>= -2e-6)
    @test all(positive_info.z .>= -2e-6)
    @test workspace_result.x ≈ info.x atol = 2e-6
    @test workspace_result.z ≈ info.z atol = 2e-6
    @test named_workspace_result.x ≈ positive_info.x atol = 2e-6
    @test named_workspace_result.z ≈ positive_info.z atol = 2e-6
    @test solver_result.x ≈ info.x atol = 2e-6
    @test solver_result.z ≈ info.z atol = 2e-6
    @test named_solver_result.x ≈ positive_info.x atol = 2e-6
    @test named_solver_result.z ≈ positive_info.z atol = 2e-6
    @test power_result.x ≈ info.x atol = 2e-6
    @test power_result.z ≈ info.z atol = 2e-6
    @test warm_info.iterations <= 2
    @test fista_step_size(A, B) ≈ 1.0
    @test 0 < power_step < fista_step_size(A, B)
end

@testset "noise-matching hyperparameter sweeps" begin
    A = Matrix{Float64}(I, 3, 3)
    b = [2.0, -1.0, 0.2]

    signed_target = norm(lasso_fista(A, b, 0.5) - b)
    signed = lasso_noise_sweep(
        A,
        b,
        signed_target;
        algorithm=:fista,
        steps=8,
        abstol=1e-9,
        reltol=1e-8,
    )

    positive_target = norm([1.5, 0.0, 0.0] - b)
    positive = nonnegative_lasso_noise_sweep(
        A,
        b,
        positive_target;
        algorithm=:admm,
        steps=8,
        rho=1.0,
        abstol=1e-9,
        reltol=1e-8,
    )

    @test signed.lambda ≈ 0.5 atol = 1e-6
    @test signed.residual ≈ signed_target atol = 2e-6
    @test signed.x ≈ lasso_fista(A, b, 0.5) atol = 2e-6
    @test length(signed.trials) <= 8
    @test signed.info.converged

    @test positive.lambda ≈ 0.5 atol = 1e-6
    @test positive.residual ≈ positive_target atol = 2e-6
    @test positive.x ≈ [1.5, 0.0, 0.0] atol = 2e-6
    @test length(positive.trials) <= 8
    @test positive.info.converged

    @test_throws ArgumentError lasso_noise_sweep(A, b, -1.0)
    @test_throws ArgumentError lasso_noise_sweep(A, b, 1.0; steps=0)
    @test_throws ArgumentError lasso_noise_sweep(A, b, 1.0; algorithm=:unknown)
    @test_throws ArgumentError lasso_noise_sweep(A, b, 1.0; lambda_high=0.0)
    @test_throws ArgumentError lasso_noise_sweep(A, b, 1.0; warm_start=false)
    @test_throws ArgumentError lasso_noise_sweep(A, b, 1.0; return_info=false)
end

@testset "cross-validation hyperparameter sweeps" begin
    base_A = Matrix{Float64}(I, 3, 3)
    A = vcat(base_A, base_A, base_A)
    b = repeat([2.0, -1.0, 0.2], 3)
    lambdas = [1.0, 0.5, 0.0]

    signed = lasso_crossvalidation(
        A,
        b;
        folds=3,
        lambdas,
        algorithm=:fista,
        abstol=1e-9,
        reltol=1e-8,
    )
    positive = nonnegative_lasso_crossvalidation(
        A,
        b;
        folds=3,
        lambdas,
        algorithm=:admm,
        rho=1.0,
        abstol=1e-9,
        reltol=1e-8,
    )

    @test signed.lambda == 0.0
    @test signed.x ≈ [2.0, -1.0, 0.2] atol = 2e-6
    @test signed.validation_error ≈ 0.0 atol = 2e-10
    @test signed.validation_error == minimum(signed.validation_errors)
    @test size(signed.fold_errors) == (length(lambdas), 3)
    @test length(signed.fold_indices) == 3
    @test all(length.(signed.fold_indices) .== 3)
    @test signed.info.converged

    @test positive.lambda == 0.0
    @test positive.x ≈ [2.0, 0.0, 0.2] atol = 2e-6
    @test positive.validation_error ≈ 1 / 3 atol = 2e-6
    @test positive.validation_error == minimum(positive.validation_errors)
    @test size(positive.fold_errors) == (length(lambdas), 3)
    @test positive.info.converged

    @test_throws ArgumentError lasso_crossvalidation(A, b; folds=1)
    @test_throws ArgumentError lasso_crossvalidation(A, b; folds=length(b) + 1)
    @test_throws ArgumentError lasso_crossvalidation(A, b; lambdas=Float64[])
    @test_throws ArgumentError lasso_crossvalidation(A, b; lambdas=[-1.0])
    @test_throws ArgumentError lasso_crossvalidation(A, b; grid_length=0)
    @test_throws ArgumentError lasso_crossvalidation(A, b; min_ratio=2.0)
    @test_throws ArgumentError lasso_crossvalidation(A, b; algorithm=:unknown)
    @test_throws ArgumentError lasso_crossvalidation(A, b; warm_start=false)
    @test_throws ArgumentError lasso_crossvalidation(A, b; return_info=false)
end

@testset "ADMM input validation" begin
    A = Matrix{Float64}(I, 2, 2)
    b = ones(2)
    workspace = LassoADMMWorkspace(A, b)
    solver = LassoADMMSolver(A, b)
    fista_workspace = LassoFISTAWorkspace(A, b)
    fista_solver = LassoFISTASolver(A, b)
    ridge_workspace = LassoRidgeFISTAWorkspace(A, A, b)
    ridge_solver = LassoRidgeFISTASolver(A, A, b)

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
    @test_throws ArgumentError lasso_ridge_fista(A, A, b, -1.0, 1.0)
    @test_throws ArgumentError lasso_ridge_fista!(ridge_workspace, 1.0, -1.0)
    @test_throws ArgumentError lasso_ridge_fista(ridge_solver, -1.0, 1.0)
    @test_throws ArgumentError LassoRidgeFISTAWorkspace(A, A, b; step=0.0)
    @test_throws ArgumentError fista_step_size(A, A; method=:unknown)
    @test_throws ArgumentError fista_step_size(A, A; method=:power, iterations=0)
    @test_throws ArgumentError lasso_admm(A, b, 1.0; rho=0.0)
    @test_throws DimensionMismatch lasso_admm(A, ones(3), 1.0)
    @test_throws DimensionMismatch lasso_admm(A, b, 1.0; x0=zeros(3))
    @test_throws DimensionMismatch lasso_ridge_fista(A, ones(3, 2), b, 1.0, 1.0)
    @test_throws DimensionMismatch lasso_ridge_fista(A, A, ones(3), 1.0, 1.0)
    @test_throws DimensionMismatch LassoRidgeFISTAWorkspace(A, A, b; z0=zeros(3))
end
