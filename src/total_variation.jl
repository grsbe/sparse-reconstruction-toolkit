export TotalVariationPDHGSolver,
    TotalVariationPDHGWorkspace,
    total_variation2d,
    total_variation_pdhg,
    total_variation_pdhg!,
    nonnegative_total_variation_pdhg,
    nonnegative_total_variation_pdhg!,
    l1_total_variation_pdhg,
    l1_total_variation_pdhg!,
    nonnegative_l1_total_variation_pdhg,
    nonnegative_l1_total_variation_pdhg!,
    tv_pdhg_step_sizes

"""
    total_variation2d(x, image_size; isotropic=true)

Return the forward-difference 2D total variation of vectorized image `x`.
`image_size` is `(height, width)` and uses Julia's column-major vectorization.
"""
function total_variation2d(x, image_size; isotropic=true)
    height, width = _validate_image_size(image_size)
    length(x) == height * width ||
        throw(DimensionMismatch("x must have length prod(image_size)"))

    T = float(eltype(x))
    value = zero(T)
    @inbounds for j in 1:width, i in 1:height
        k = i + (j - 1) * height
        gx = i < height ? x[k + 1] - x[k] : zero(T)
        gy = j < width ? x[k + height] - x[k] : zero(T)
        if isotropic
            value += hypot(gx, gy)
        else
            value += abs(gx) + abs(gy)
        end
    end
    return value
end

# Reuse this solver when A, b, image size, and PDHG step sizes stay fixed.
struct TotalVariationPDHGSolver{W}
    workspace::W
end

TotalVariationPDHGSolver(A, b, image_size; kwargs...) =
    TotalVariationPDHGSolver(TotalVariationPDHGWorkspace(A, b, image_size; kwargs...))

mutable struct TotalVariationPDHGWorkspace{M,B,V,Q,P,T}
    A::M
    b::B
    image_size::Tuple{Int,Int}
    x::V
    x_previous::V
    x_bar::V
    gradient::V
    prox_input::V
    data_dual::Q
    data_dual_previous::Q
    tv_dual_x::P
    tv_dual_x_previous::P
    tv_dual_y::P
    tv_dual_y_previous::P
    residual::Q
    primal_step::T
    dual_step::T
    theta::T
end

function TotalVariationPDHGWorkspace(
    A,
    b,
    image_size;
    primal_step=nothing,
    dual_step=nothing,
    theta=1.0,
    x0=nothing,
    step_method=:exact,
    power_iterations=20,
    safety=0.99,
)
    _validate_problem(A, b)
    size(A, 2) == prod(_validate_image_size(image_size)) ||
        throw(DimensionMismatch("A must have one column per image pixel"))
    _validate_positive(theta, "theta")

    x = _initial_iterate(A, b, x0)
    if isnothing(primal_step) || isnothing(dual_step)
        default_primal, default_dual = tv_pdhg_step_sizes(
            A;
            method=step_method,
            iterations=power_iterations,
            safety,
        )
        primal_step = isnothing(primal_step) ? default_primal : float(primal_step)
        dual_step = isnothing(dual_step) ? default_dual : float(dual_step)
    else
        primal_step = float(primal_step)
        dual_step = float(dual_step)
    end
    _validate_positive(primal_step, "primal_step")
    _validate_positive(dual_step, "dual_step")

    height, width = _validate_image_size(image_size)
    T = eltype(x)
    return TotalVariationPDHGWorkspace(
        A,
        b,
        (height, width),
        x,
        similar(x),
        copy(x),
        similar(x),
        similar(x),
        zeros(T, length(b)),
        zeros(T, length(b)),
        zeros(T, height, width),
        zeros(T, height, width),
        zeros(T, height, width),
        zeros(T, height, width),
        zeros(T, length(b)),
        primal_step,
        dual_step,
        float(theta),
    )
end

"""
    tv_pdhg_step_sizes(A; method=:exact, iterations=20, safety=0.99)

Return equal primal and dual PDHG step sizes for the operator `[A; nabla]`.
The 2D forward-difference contribution uses the bound `norm(nabla)^2 <= 8`.
"""
function tv_pdhg_step_sizes(A; method=:exact, iterations=20, safety=0.99)
    ndims(A) == 2 || throw(DimensionMismatch("A must be a matrix"))
    _validate_positive(safety, "safety")
    safety < 1 || throw(ArgumentError("safety must be less than 1"))

    data_norm2 = if method === :exact
        opnorm(A)^2
    elseif method === :power
        iterations isa Integer && iterations > 0 ||
            throw(ArgumentError("iterations must be a positive integer"))
        _power_lipschitz(A, iterations)
    else
        throw(ArgumentError("method must be :exact or :power"))
    end
    operator_norm = sqrt(max(data_norm2 + 8, eps(float(data_norm2))))
    step = float(safety) / operator_norm
    return step, step
end

function total_variation_pdhg(A, b, image_size, lambda_tv; kwargs...)
    _validate_nonnegative(lambda_tv, "lambda_tv")
    return _total_variation_pdhg(A, b, image_size, lambda_tv, zero(float(lambda_tv)); kwargs...)
end

function nonnegative_total_variation_pdhg(A, b, image_size, lambda_tv; kwargs...)
    _validate_nonnegative(lambda_tv, "lambda_tv")
    return _total_variation_pdhg(
        A,
        b,
        image_size,
        lambda_tv,
        zero(float(lambda_tv));
        nonnegative=true,
        kwargs...,
    )
end

function l1_total_variation_pdhg(A, b, image_size, lambda_l1, lambda_tv=0.0; kwargs...)
    _validate_nonnegative(lambda_l1, "lambda_l1")
    _validate_nonnegative(lambda_tv, "lambda_tv")
    return _total_variation_pdhg(A, b, image_size, lambda_tv, lambda_l1; kwargs...)
end

function nonnegative_l1_total_variation_pdhg(
    A,
    b,
    image_size,
    lambda_l1,
    lambda_tv=0.0;
    kwargs...,
)
    _validate_nonnegative(lambda_l1, "lambda_l1")
    _validate_nonnegative(lambda_tv, "lambda_tv")
    return _total_variation_pdhg(
        A,
        b,
        image_size,
        lambda_tv,
        lambda_l1;
        nonnegative=true,
        kwargs...,
    )
end

function total_variation_pdhg(solver::TotalVariationPDHGSolver, lambda_tv; kwargs...)
    return total_variation_pdhg!(solver.workspace, lambda_tv; kwargs...)
end

function nonnegative_total_variation_pdhg(
    solver::TotalVariationPDHGSolver,
    lambda_tv;
    kwargs...,
)
    return nonnegative_total_variation_pdhg!(solver.workspace, lambda_tv; kwargs...)
end

function l1_total_variation_pdhg(
    solver::TotalVariationPDHGSolver,
    lambda_l1,
    lambda_tv=0.0;
    kwargs...,
)
    return l1_total_variation_pdhg!(solver.workspace, lambda_l1, lambda_tv; kwargs...)
end

function nonnegative_l1_total_variation_pdhg(
    solver::TotalVariationPDHGSolver,
    lambda_l1,
    lambda_tv=0.0;
    kwargs...,
)
    return nonnegative_l1_total_variation_pdhg!(
        solver.workspace,
        lambda_l1,
        lambda_tv;
        kwargs...,
    )
end

function total_variation_pdhg!(
    workspace::TotalVariationPDHGWorkspace,
    lambda_tv;
    warm_start=false,
    kwargs...,
)
    _validate_nonnegative(lambda_tv, "lambda_tv")
    return _total_variation_pdhg!(
        workspace,
        lambda_tv,
        zero(float(lambda_tv));
        warm_start,
        kwargs...,
    )
end

function nonnegative_total_variation_pdhg!(
    workspace::TotalVariationPDHGWorkspace,
    lambda_tv;
    warm_start=false,
    kwargs...,
)
    _validate_nonnegative(lambda_tv, "lambda_tv")
    return _total_variation_pdhg!(
        workspace,
        lambda_tv,
        zero(float(lambda_tv));
        nonnegative=true,
        warm_start,
        kwargs...,
    )
end

function l1_total_variation_pdhg!(
    workspace::TotalVariationPDHGWorkspace,
    lambda_l1,
    lambda_tv=0.0;
    warm_start=false,
    kwargs...,
)
    _validate_nonnegative(lambda_l1, "lambda_l1")
    _validate_nonnegative(lambda_tv, "lambda_tv")
    return _total_variation_pdhg!(
        workspace,
        lambda_tv,
        lambda_l1;
        warm_start,
        kwargs...,
    )
end

function nonnegative_l1_total_variation_pdhg!(
    workspace::TotalVariationPDHGWorkspace,
    lambda_l1,
    lambda_tv=0.0;
    warm_start=false,
    kwargs...,
)
    _validate_nonnegative(lambda_l1, "lambda_l1")
    _validate_nonnegative(lambda_tv, "lambda_tv")
    return _total_variation_pdhg!(
        workspace,
        lambda_tv,
        lambda_l1;
        nonnegative=true,
        warm_start,
        kwargs...,
    )
end

function _total_variation_pdhg(
    A,
    b,
    image_size,
    lambda_tv,
    lambda_l1;
    x0=nothing,
    primal_step=nothing,
    dual_step=nothing,
    theta=1.0,
    step_method=:exact,
    power_iterations=20,
    safety=0.99,
    kwargs...,
)
    workspace = TotalVariationPDHGWorkspace(
        A,
        b,
        image_size;
        x0,
        primal_step,
        dual_step,
        theta,
        step_method,
        power_iterations,
        safety,
    )
    return _total_variation_pdhg!(
        workspace,
        lambda_tv,
        lambda_l1;
        warm_start=true,
        kwargs...,
    )
end

function _total_variation_pdhg!(
    workspace::TotalVariationPDHGWorkspace,
    lambda_tv,
    lambda_l1;
    nonnegative=false,
    isotropic=true,
    warm_start=false,
    abstol=1e-6,
    reltol=1e-4,
    maxiter=10_000,
    return_info=false,
)
    _validate_nonnegative(lambda_tv, "lambda_tv")
    _validate_nonnegative(lambda_l1, "lambda_l1")
    _validate_nonnegative(abstol, "abstol")
    _validate_nonnegative(reltol, "reltol")
    maxiter isa Integer && maxiter > 0 ||
        throw(ArgumentError("maxiter must be a positive integer"))

    x = workspace.x
    x_previous = workspace.x_previous
    x_bar = workspace.x_bar
    gradient = workspace.gradient
    prox_input = workspace.prox_input
    data_dual = workspace.data_dual
    data_dual_previous = workspace.data_dual_previous
    tv_dual_x = workspace.tv_dual_x
    tv_dual_x_previous = workspace.tv_dual_x_previous
    tv_dual_y = workspace.tv_dual_y
    tv_dual_y_previous = workspace.tv_dual_y_previous
    residual = workspace.residual
    tau = workspace.primal_step
    sigma = workspace.dual_step
    theta = workspace.theta
    root_n = sqrt(length(x))

    if !warm_start
        fill!(x, zero(eltype(x)))
        fill!(data_dual, zero(eltype(data_dual)))
        fill!(data_dual_previous, zero(eltype(data_dual_previous)))
        fill!(tv_dual_x, zero(eltype(tv_dual_x)))
        fill!(tv_dual_x_previous, zero(eltype(tv_dual_x_previous)))
        fill!(tv_dual_y, zero(eltype(tv_dual_y)))
        fill!(tv_dual_y_previous, zero(eltype(tv_dual_y_previous)))
    end
    copyto!(x_bar, x)

    iterate_change = Inf
    converged = false
    iterations = maxiter

    for iteration in 1:maxiter
        copyto!(x_previous, x)
        copyto!(data_dual_previous, data_dual)
        copyto!(tv_dual_x_previous, tv_dual_x)
        copyto!(tv_dual_y_previous, tv_dual_y)

        mul!(residual, workspace.A, x_bar)
        @. residual = residual - workspace.b
        @. data_dual = (data_dual + sigma * residual) / (one(sigma) + sigma)

        _tv_dual_step_project!(
            tv_dual_x,
            tv_dual_y,
            x_bar,
            workspace.image_size,
            sigma,
            lambda_tv,
            isotropic,
        )

        mul!(gradient, adjoint(workspace.A), data_dual)
        _add_tv_adjoint!(gradient, tv_dual_x, tv_dual_y, workspace.image_size)
        @. prox_input = x - tau * gradient
        _prox_l1_nonnegative!(x, prox_input, tau * lambda_l1, nonnegative)

        @. prox_input = x - x_previous
        iterate_change = sqrt(
            norm(prox_input)^2 +
            _squared_difference(data_dual, data_dual_previous) +
            _squared_difference(tv_dual_x, tv_dual_x_previous) +
            _squared_difference(tv_dual_y, tv_dual_y_previous),
        )
        current_norm = sqrt(
            norm(x)^2 + norm(data_dual)^2 + norm(tv_dual_x)^2 + norm(tv_dual_y)^2,
        )
        previous_norm = sqrt(
            norm(x_previous)^2 +
            norm(data_dual_previous)^2 +
            norm(tv_dual_x_previous)^2 +
            norm(tv_dual_y_previous)^2,
        )
        iterate_tolerance =
            root_n * abstol + reltol * max(current_norm, previous_norm)
        if iterate_change <= iterate_tolerance
            converged = true
            iterations = iteration
            break
        end

        @. x_bar = x + theta * (x - x_previous)
    end

    if return_info
        return (
            x=x,
            converged=converged,
            iterations=iterations,
            iterate_change=iterate_change,
            primal_step=tau,
            dual_step=sigma,
            lambda_tv=lambda_tv,
            lambda_l1=lambda_l1,
            nonnegative=nonnegative,
        )
    end

    return x
end

function _tv_dual_step_project!(
    dual_x,
    dual_y,
    x,
    image_size,
    sigma,
    lambda_tv,
    isotropic,
)
    height, width = image_size
    T = eltype(dual_x)
    radius = T(lambda_tv)
    @inbounds for j in 1:width, i in 1:height
        k = i + (j - 1) * height
        gx = i < height ? x[k + 1] - x[k] : zero(T)
        gy = j < width ? x[k + height] - x[k] : zero(T)
        px = dual_x[i, j] + sigma * gx
        py = dual_y[i, j] + sigma * gy
        if isotropic
            scale = iszero(radius) ? T(Inf) : max(one(T), hypot(px, py) / radius)
            dual_x[i, j] = px / scale
            dual_y[i, j] = py / scale
        else
            dual_x[i, j] = clamp(px, -radius, radius)
            dual_y[i, j] = clamp(py, -radius, radius)
        end
    end
    return dual_x, dual_y
end

function _add_tv_adjoint!(gradient, dual_x, dual_y, image_size)
    height, width = image_size
    @inbounds for j in 1:width, i in 1:height
        k = i + (j - 1) * height
        value = zero(eltype(gradient))
        if i > 1
            value += dual_x[i - 1, j]
        end
        if i < height
            value -= dual_x[i, j]
        end
        if j > 1
            value += dual_y[i, j - 1]
        end
        if j < width
            value -= dual_y[i, j]
        end
        gradient[k] += value
    end
    return gradient
end

function _squared_difference(x, y)
    value = zero(float(promote_type(eltype(x), eltype(y))))
    @inbounds for index in eachindex(x, y)
        value += abs2(x[index] - y[index])
    end
    return value
end

function _prox_l1_nonnegative!(x, prox_input, threshold, nonnegative)
    T = eltype(x)
    threshold = T(threshold)
    if nonnegative
        @. x = max(prox_input - threshold, zero(T))
    else
        @. x = sign(prox_input) * max(abs(prox_input) - threshold, zero(T))
    end
    return x
end

function _validate_image_size(image_size)
    length(image_size) == 2 ||
        throw(DimensionMismatch("image_size must be a tuple or vector of length 2"))
    height, width = Int(image_size[1]), Int(image_size[2])
    height > 0 && width > 0 ||
        throw(DimensionMismatch("image_size entries must be positive"))
    return height, width
end
