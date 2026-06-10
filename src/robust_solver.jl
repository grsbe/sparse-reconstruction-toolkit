export SoftminLassoSolver,
    SoftminLassoWorkspace,
    robust_lasso,
    softmin_lasso,
    softmin_lasso!,
    softmin_lasso_gradient,
    softmin_lasso_loss,
    softmin_lasso_loss_gradient,
    softmin_lasso_objective,
    softmin_lasso_step_size,
    softmin_lasso_weights

"""
    softmin_lasso(A, y, lambda; tau, kwargs...)

Solve sparse regression where each measurement has multiple possible perturbed
design rows and only one row per measurement needs to explain the target.

The data term is the smooth soft-min relaxation of
`sum_i min_k (dot(A[i, k, :], x) - y[i])^2`, with `A` shaped `(m, K, n)`.
The full objective adds `lambda * norm(x, 1)` and is minimized by proximal
gradient steps using the L1 prox from ProximalOperators.

`SoftminLassoWorkspace` caches each perturbation slice as a contiguous matrix by
default because the solver repeatedly applies `A[:, k, :]` and its adjoint. Pass
`copy_slices=false` to save memory when this copy is too expensive.
"""
function softmin_lasso(A, y, lambda; tau, step=nothing, x0=nothing, kwargs...)
    workspace = SoftminLassoWorkspace(A, y; step, x0)
    return softmin_lasso!(workspace, lambda; tau, warm_start=true, kwargs...)
end

robust_lasso(args...; kwargs...) = softmin_lasso(args...; kwargs...)

struct SoftminLassoSolver{W}
    workspace::W
end

SoftminLassoSolver(A, y; kwargs...) =
    SoftminLassoSolver(SoftminLassoWorkspace(A, y; kwargs...))

mutable struct SoftminLassoWorkspace{AType,YType,V,T,M,R,W}
    A::AType
    y::YType
    x::V
    x_previous::V
    gradient::V
    prox_input::V
    step::T
    slices::Vector{M}
    residuals::R
    weights::R
    weighted_residual::W
end

function SoftminLassoWorkspace(A, y; step=nothing, x0=nothing, copy_slices=true)
    _validate_softmin_problem(A, y)
    step = isnothing(step) ? softmin_lasso_step_size(A) : float(step)
    _validate_positive(step, "step")
    x = _softmin_initial_iterate(A, y, x0)
    m, K, _ = size(A)
    slices = if copy_slices
        [copy(view(A, :, k, :)) for k in 1:K]
    else
        [view(A, :, k, :) for k in 1:K]
    end
    residuals = zeros(eltype(x), m, K)
    weights = similar(residuals)
    weighted_residual = zeros(eltype(x), m)
    return SoftminLassoWorkspace(
        A,
        y,
        x,
        similar(x),
        similar(x),
        similar(x),
        step,
        slices,
        residuals,
        weights,
        weighted_residual,
    )
end

function softmin_lasso(solver::SoftminLassoSolver, lambda; kwargs...)
    return softmin_lasso!(solver.workspace, lambda; kwargs...)
end

function softmin_lasso!(
    workspace::SoftminLassoWorkspace,
    lambda;
    tau,
    warm_start=false,
    abstol=1e-6,
    reltol=1e-4,
    maxiter=10_000,
    backtracking=true,
    backtrack_factor=0.5,
    max_backtracks=30,
    accept_tol=1e-12,
    return_info=false,
    history=return_info,
)
    _validate_nonnegative(lambda, "lambda")
    _validate_nonnegative(abstol, "abstol")
    _validate_nonnegative(reltol, "reltol")
    maxiter isa Integer && maxiter > 0 ||
        throw(ArgumentError("maxiter must be a positive integer"))
    _validate_positive(backtrack_factor, "backtrack_factor")
    backtrack_factor < 1 ||
        throw(ArgumentError("backtrack_factor must be less than 1"))
    max_backtracks isa Integer && max_backtracks >= 0 ||
        throw(ArgumentError("max_backtracks must be a nonnegative integer"))
    _validate_nonnegative(accept_tol, "accept_tol")

    tau_schedule = _softmin_tau_schedule(tau)
    nonsmooth_term = NormL1(lambda)

    if !warm_start
        fill!(workspace.x, zero(eltype(workspace.x)))
    end

    stage_infos = history ? Vector{Any}(undef, length(tau_schedule)) : nothing
    last_info = nothing
    for (stage, tau_value) in pairs(tau_schedule)
        last_info = _softmin_lasso_single_tau!(
            workspace,
            nonsmooth_term,
            lambda,
            tau_value;
            abstol,
            reltol,
            maxiter,
            backtracking,
            backtrack_factor,
            max_backtracks,
            accept_tol,
            return_info=(return_info || history),
            history,
        )
        if history
            stage_infos[stage] = last_info
        end
    end

    if return_info
        return _softmin_return_info(workspace, lambda, tau_schedule[end], last_info, stage_infos)
    end

    return workspace.x
end

"""
    softmin_lasso_step_size(A; safety=10.0)

Return a conservative initial proximal-gradient step size for `softmin_lasso`.
The estimate uses a row-norm curvature bound for arrays shaped `(m, K, n)` and
is intended to reduce expensive backtracking from an arbitrary `step=1.0`.
"""
function softmin_lasso_step_size(A; safety=10.0)
    ndims(A) == 3 || throw(ArgumentError("A must be a 3-dimensional array"))
    _validate_positive(safety, "safety")

    m, K, n = size(A)
    T = float(eltype(A))
    row_bound = zero(T)

    @inbounds for i in 1:m
        max_row_norm = zero(T)
        for k in 1:K
            row_norm = zero(T)
            for j in 1:n
                row_norm += abs2(T(A[i, k, j]))
            end
            max_row_norm = max(max_row_norm, row_norm)
        end
        row_bound += max_row_norm
    end

    lipschitz_bound = 2 * T(safety) * row_bound
    return iszero(lipschitz_bound) ? one(T) : inv(lipschitz_bound)
end

function softmin_lasso_loss(A, y, x, tau)
    loss, _ = softmin_lasso_loss_gradient(A, y, x, tau)
    return loss
end

function softmin_lasso_gradient(A, y, x, tau)
    _, gradient = softmin_lasso_loss_gradient(A, y, x, tau)
    return gradient
end

function softmin_lasso_objective(A, y, x, lambda, tau)
    _validate_nonnegative(lambda, "lambda")
    return softmin_lasso_loss(A, y, x, tau) + lambda * norm(x, 1)
end

function softmin_lasso_loss_gradient(A, y, x, tau)
    _validate_softmin_problem(A, y)
    _validate_softmin_x(A, x)
    _validate_positive(tau, "tau")

    m, K, n = size(A)
    T = float(promote_type(eltype(A), eltype(y), eltype(x), typeof(tau)))
    gradient = zeros(T, n)
    residuals = Vector{T}(undef, K)
    scores = Vector{T}(undef, K)
    weights = Vector{T}(undef, K)
    loss = zero(T)
    tau = T(tau)

    @inbounds for i in 1:m
        for k in 1:K
            prediction = zero(T)
            for j in 1:n
                prediction += A[i, k, j] * x[j]
            end
            residuals[k] = prediction - y[i]
            scores[k] = -(residuals[k]^2) / tau
        end

        score_max = maximum(scores)
        normalizer = zero(T)
        for k in 1:K
            weights[k] = exp(scores[k] - score_max)
            normalizer += weights[k]
        end

        loss += -tau * (score_max + log(normalizer) - log(T(K)))

        for k in 1:K
            weight = weights[k] / normalizer
            coefficient = 2 * weight * residuals[k]
            for j in 1:n
                gradient[j] += coefficient * A[i, k, j]
            end
        end
    end

    return loss, gradient
end

function softmin_lasso_weights(A, y, x, tau)
    _validate_softmin_problem(A, y)
    _validate_softmin_x(A, x)
    _validate_positive(tau, "tau")

    m, K, n = size(A)
    T = float(promote_type(eltype(A), eltype(y), eltype(x), typeof(tau)))
    result = zeros(T, m, K)
    scores = Vector{T}(undef, K)
    tau = T(tau)

    @inbounds for i in 1:m
        for k in 1:K
            prediction = zero(T)
            for j in 1:n
                prediction += A[i, k, j] * x[j]
            end
            residual = prediction - y[i]
            scores[k] = -(residual^2) / tau
        end

        score_max = maximum(scores)
        normalizer = zero(T)
        for k in 1:K
            result[i, k] = exp(scores[k] - score_max)
            normalizer += result[i, k]
        end
        for k in 1:K
            result[i, k] /= normalizer
        end
    end

    return result
end


function _softmin_loss_gradient!(workspace::SoftminLassoWorkspace, tau)
    loss = _softmin_residuals_weights!(workspace, tau)
    gradient = workspace.gradient
    weighted_residual = workspace.weighted_residual
    residuals = workspace.residuals
    weights = workspace.weights
    T = eltype(gradient)

    fill!(gradient, zero(T))
    @views for k in axes(residuals, 2)
        @. weighted_residual = 2 * weights[:, k] * residuals[:, k]
        mul!(gradient, adjoint(workspace.slices[k]), weighted_residual, one(T), one(T))
    end

    return loss
end

function _softmin_loss!(workspace::SoftminLassoWorkspace, tau)
    return _softmin_residuals_weights!(workspace, tau)
end

function _softmin_weights(workspace::SoftminLassoWorkspace, tau)
    _softmin_residuals_weights!(workspace, tau)
    return copy(workspace.weights)
end

function _softmin_residuals_weights!(workspace::SoftminLassoWorkspace, tau)
    _validate_positive(tau, "tau")

    residuals = workspace.residuals
    weights = workspace.weights
    x = workspace.x
    y = workspace.y
    T = eltype(residuals)
    tau = T(tau)
    m, K = size(residuals)

    @views for k in 1:K
        mul!(residuals[:, k], workspace.slices[k], x)
        @. residuals[:, k] = residuals[:, k] - y
        @. weights[:, k] = -(residuals[:, k]^2) / tau
    end

    loss = zero(T)
    @inbounds for i in 1:m
        score_max = weights[i, 1]
        for k in 2:K
            score_max = max(score_max, weights[i, k])
        end

        normalizer = zero(T)
        for k in 1:K
            weights[i, k] = exp(weights[i, k] - score_max)
            normalizer += weights[i, k]
        end

        loss += -tau * (score_max + log(normalizer) - log(T(K)))

        inv_normalizer = inv(normalizer)
        for k in 1:K
            weights[i, k] *= inv_normalizer
        end
    end

    return loss
end

function _softmin_lasso_single_tau!(
    workspace,
    nonsmooth_term,
    lambda,
    tau;
    abstol,
    reltol,
    maxiter,
    backtracking,
    backtrack_factor,
    max_backtracks,
    accept_tol,
    return_info,
    history,
)
    A = workspace.A
    y = workspace.y
    x = workspace.x
    x_previous = workspace.x_previous
    gradient = workspace.gradient
    prox_input = workspace.prox_input
    root_n = sqrt(length(x))
    step = workspace.step

    objective_history = history ? Vector{eltype(x)}() : nothing
    data_loss_history = history ? Vector{eltype(x)}() : nothing
    l1_history = history ? Vector{eltype(x)}() : nothing
    residual_history = history ? Vector{eltype(x)}() : nothing
    backtrack_history = history ? Int[] : nothing
    total_backtracks = 0

    data_loss = _softmin_loss_gradient!(workspace, tau)
    objective = data_loss + lambda * norm(x, 1)

    converged = false
    iterations = maxiter
    iterate_change = Inf
    proximal_residual = Inf

    for iteration in 1:maxiter
        copyto!(x_previous, x)

        trial_step = step
        trial_objective = objective
        trial_data_loss = data_loss
        backtracks = 0
        while true
            @. prox_input = x_previous - trial_step * gradient
            _prox_nonsmooth!(x, nonsmooth_term, prox_input, trial_step)
            trial_data_loss = _softmin_loss!(workspace, tau)
            trial_objective = trial_data_loss + lambda * norm(x, 1)

            if !backtracking ||
               trial_objective <= objective + accept_tol ||
               backtracks >= max_backtracks
                break
            end

            trial_step *= backtrack_factor
            backtracks += 1
        end

        step = trial_step
        total_backtracks += backtracks
        objective = trial_objective
        data_loss = trial_data_loss
        @. prox_input = x - x_previous
        iterate_change = norm(prox_input)
        proximal_residual = iterate_change / step

        if history
            push!(objective_history, objective)
            push!(data_loss_history, data_loss)
            push!(l1_history, norm(x, 1))
            push!(residual_history, proximal_residual)
            push!(backtrack_history, backtracks)
        end

        iterate_tolerance =
            root_n * abstol + reltol * max(norm(x), norm(x_previous))
        if iterate_change <= iterate_tolerance
            converged = true
            iterations = iteration
            break
        end

        data_loss = _softmin_loss_gradient!(workspace, tau)
    end

    workspace.step = step

    if return_info
        return (
            tau=tau,
            converged=converged,
            iterations=iterations,
            iterate_change=iterate_change,
            proximal_residual=proximal_residual,
            objective=objective,
            data_loss=data_loss,
            l1_norm=norm(x, 1),
            step=step,
            backtracks=total_backtracks,
            backtrack_counts=backtrack_history,
            objective_values=objective_history,
            data_loss_values=data_loss_history,
            l1_norms=l1_history,
            proximal_residuals=residual_history,
        )
    end

    return nothing
end

function _softmin_return_info(workspace, lambda, tau, last_info, stage_infos)
    weights = _softmin_weights(workspace, tau)
    return (
        x=workspace.x,
        converged=last_info.converged,
        iterations=last_info.iterations,
        iterate_change=last_info.iterate_change,
        proximal_residual=last_info.proximal_residual,
        objective=last_info.objective,
        data_loss=last_info.data_loss,
        l1_norm=last_info.l1_norm,
        step=last_info.step,
        backtracks=last_info.backtracks,
        backtrack_counts=last_info.backtrack_counts,
        tau=tau,
        lambda=lambda,
        weights=weights,
        objective_values=last_info.objective_values,
        data_loss_values=last_info.data_loss_values,
        l1_norms=last_info.l1_norms,
        proximal_residuals=last_info.proximal_residuals,
        stages=stage_infos,
    )
end

function _softmin_tau_schedule(tau)
    if tau isa Number
        _validate_positive(tau, "tau")
        return [float(tau)]
    end

    values = float.(collect(tau))
    !isempty(values) || throw(ArgumentError("tau schedule must be nonempty"))
    for value in values
        _validate_positive(value, "tau")
    end
    return values
end

function _softmin_initial_iterate(A, y, x0)
    n = size(A, 3)
    T = float(promote_type(eltype(A), eltype(y)))

    if isnothing(x0)
        return zeros(T, n)
    end

    length(x0) == n ||
        throw(ArgumentError("x0 must have length $(n)"))
    return T.(x0)
end

function _validate_softmin_problem(A, y)
    ndims(A) == 3 || throw(ArgumentError("A must be a 3-dimensional array"))
    ndims(y) == 1 || throw(ArgumentError("y must be a vector"))
    length(y) == size(A, 1) ||
        throw(ArgumentError("length(y) must equal size(A, 1)"))
    size(A, 2) > 0 || throw(ArgumentError("A must contain at least one perturbation"))
    size(A, 3) > 0 || throw(ArgumentError("A must contain at least one feature"))
    return nothing
end

function _validate_softmin_x(A, x)
    ndims(x) == 1 || throw(ArgumentError("x must be a vector"))
    length(x) == size(A, 3) ||
        throw(ArgumentError("length(x) must equal size(A, 3)"))
    return nothing
end
