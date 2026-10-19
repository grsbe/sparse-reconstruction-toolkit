using LinearAlgebra

export GPUFISTASolver, gpu_lasso_fista, gpu_nonnegative_lasso_fista

"""
    GPUFISTASolver(A, b; step=nothing, x0=nothing, power_iterations=50)

Reusable FISTA solver for GPU-resident A and b. The arrays must support
mul!, norm, adjoint, and broadcasting (for example CUDA.jl dense CuArrays
or a cuSPARSE matrix with a CuArray right-hand side). Scratch arrays are
allocated with similar(b), so they remain on the same device.

The default step uses an approximate power iteration on the device. Pass a
known conservative step if a certified Lipschitz bound is needed. Keep A
and b fixed while reusing this solver for multiple values of lambda.
"""
mutable struct GPUFISTASolver{M,B,V,R,T}
    A::M
    b::B
    x::V
    y::V
    x_previous::V
    residual::R
    gradient::V
    prox_input::V
    step::T
end

function GPUFISTASolver(A, b; step=nothing, x0=nothing, power_iterations=50)
    _validate_problem(A, b)
    eltype(A) == eltype(b) ||
        throw(ArgumentError("A and b must have the same element type for GPU matrix-vector multiplication"))
    T = eltype(A)
    T <: AbstractFloat ||
        throw(ArgumentError("A and b must have real floating-point elements"))
    x = similar(b, T, size(A, 2))
    if isnothing(x0)
        fill!(x, zero(T))
    else
        length(x0) == length(x) ||
            throw(DimensionMismatch("x0 must have one entry per column of A"))
        copyto!(x, x0)
    end
    step = isnothing(step) ?
           _gpu_fista_power_step(A, x, b, power_iterations) : float(step)
    _validate_positive(step, "step")
    residual = similar(b, T, length(b))
    return GPUFISTASolver(
        A, b, x, copy(x), similar(x), residual, similar(x), similar(x), step,
    )
end

function _gpu_fista_power_step(A, template_x, template_b, iterations)
    iterations isa Integer && iterations > 0 ||
        throw(ArgumentError("power_iterations must be a positive integer"))
    T = eltype(template_x)
    x = similar(template_x)
    fill!(x, inv(sqrt(T(length(x)))))
    Ax = similar(template_b, T, length(template_b))
    AtAx = similar(template_x)
    for _ in 1:iterations
        mul!(Ax, A, x)
        mul!(AtAx, adjoint(A), Ax)
        scale = norm(AtAx)
        iszero(scale) && return one(T)
        @. x = AtAx / scale
    end
    mul!(Ax, A, x)
    estimate = norm(Ax)^2
    return iszero(estimate) ? one(T) : inv(T(1.05) * estimate)
end

"""
    gpu_lasso_fista(A, b, lambda; kwargs...)
    gpu_lasso_fista(solver::GPUFISTASolver, lambda; kwargs...)

Solve 0.5*norm(A*x-b)^2 + lambda*norm(x,1) on the device holding A and
b. gpu_nonnegative_lasso_fista additionally constrains x >= 0.
The result remains on that device. Reuse a solver for repeated solves; pass
warm_start=true to continue from the previous coefficients. Convergence is
checked every check_every iterations (default 10) to reduce device synchronization.
"""
function gpu_lasso_fista(
    A, b, lambda; step=nothing, x0=nothing, power_iterations=50,
    abstol=1e-6, reltol=1e-4, maxiter=10_000, check_every=10,
    return_info=false,
)
    solver = GPUFISTASolver(A, b; step, x0, power_iterations)
    return _gpu_lasso_fista!(
        solver, lambda, false; warm_start=true, abstol, reltol, maxiter,
        check_every, return_info,
    )
end

function gpu_nonnegative_lasso_fista(
    A, b, lambda; step=nothing, x0=nothing, power_iterations=50,
    abstol=1e-6, reltol=1e-4, maxiter=10_000, check_every=10,
    return_info=false,
)
    solver = GPUFISTASolver(A, b; step, x0, power_iterations)
    return _gpu_lasso_fista!(
        solver, lambda, true; warm_start=true, abstol, reltol, maxiter,
        check_every, return_info,
    )
end

gpu_lasso_fista(solver::GPUFISTASolver, lambda; kwargs...) =
    _gpu_lasso_fista!(solver, lambda, false; kwargs...)

gpu_nonnegative_lasso_fista(solver::GPUFISTASolver, lambda; kwargs...) =
    _gpu_lasso_fista!(solver, lambda, true; kwargs...)

function _gpu_lasso_fista!(
    solver::GPUFISTASolver, lambda, nonnegative;
    warm_start=false, abstol=1e-6, reltol=1e-4, maxiter=10_000,
    check_every=10, return_info=false,
)
    _validate_nonnegative(lambda, "lambda")
    _validate_nonnegative(abstol, "abstol")
    _validate_nonnegative(reltol, "reltol")
    maxiter isa Integer && maxiter > 0 ||
        throw(ArgumentError("maxiter must be a positive integer"))
    check_every isa Integer && check_every > 0 ||
        throw(ArgumentError("check_every must be a positive integer"))
    x, y, x_previous = solver.x, solver.y, solver.x_previous
    residual, gradient, prox_input =
        solver.residual, solver.gradient, solver.prox_input
    step = solver.step
    if !warm_start
        fill!(x, zero(eltype(x)))
    end
    copyto!(y, x)
    root_n = sqrt(length(x))
    t = one(step)
    iterate_change = Inf
    converged = false
    iterations = maxiter
    threshold = step * lambda
    zero_x = zero(eltype(x))

    for iteration in 1:maxiter
        copyto!(x_previous, x)
        mul!(residual, solver.A, y)
        @. residual = residual - solver.b
        mul!(gradient, adjoint(solver.A), residual)
        @. prox_input = y - step * gradient
        if nonnegative
            @. x = max(prox_input - threshold, zero_x)
        else
            @. x = sign(prox_input) *
                   max(abs(prox_input) - threshold, zero_x)
        end

        if iteration % check_every == 0 || iteration == maxiter
            @. prox_input = x - x_previous
            iterate_change = norm(prox_input)
            tolerance = root_n * abstol + reltol * max(norm(x), norm(x_previous))
            if iterate_change <= tolerance
                converged = true
                iterations = iteration
                break
            end
        end

        t_next = (one(t) + sqrt(one(t) + 4 * t^2)) / 2
        momentum = (t - one(t)) / t_next
        @. y = x + momentum * (x - x_previous)
        t = t_next
    end
    return return_info ?
           (x=x, converged=converged, iterations=iterations,
            iterate_change=iterate_change, step=step) : x
end
