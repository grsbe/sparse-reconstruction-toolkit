export LaplacianFISTASolver,
    LaplacianFISTAWorkspace,
    laplacian2d,
    laplacian_lasso_fista,
    laplacian_lasso_fista!,
    nonnegative_laplacian_lasso_fista,
    nonnegative_laplacian_lasso_fista!,
    laplacian_lasso_step_size

"""
    laplacian2d(x, image_size)

Return the 5-point graph Laplacian of vectorized image `x`. Boundary pixels use
only existing neighbors, so constant images have zero Laplacian.
"""
function laplacian2d(x, image_size)
    height, width = _validate_image_size(image_size)
    length(x) == height * width ||
        throw(DimensionMismatch("x must have length prod(image_size)"))

    result = similar(x, float(eltype(x)))
    return _laplacian2d!(result, x, (height, width))
end

# Reuse this solver when A, b, image size, and the FISTA step size stay fixed.
struct LaplacianFISTASolver{W}
    workspace::W
end

LaplacianFISTASolver(A, b, image_size; kwargs...) =
    LaplacianFISTASolver(LaplacianFISTAWorkspace(A, b, image_size; kwargs...))

mutable struct LaplacianFISTAWorkspace{M,B,V,R,T}
    A::M
    b::B
    image_size::Tuple{Int,Int}
    x::V
    y::V
    x_previous::V
    residual::R
    gradient::V
    prox_input::V
    laplacian::V
    bilaplacian::V
    step::T
end

function LaplacianFISTAWorkspace(
    A,
    b,
    image_size;
    step=nothing,
    x0=nothing,
    lambda_laplacian=1.0,
    step_method=:exact,
    power_iterations=20,
    safety=1.05,
)
    _validate_problem(A, b)
    size(A, 2) == prod(_validate_image_size(image_size)) ||
        throw(DimensionMismatch("A must have one column per image pixel"))
    _validate_nonnegative(lambda_laplacian, "lambda_laplacian")

    x = _initial_iterate(A, b, x0)
    step = isnothing(step) ?
           laplacian_lasso_step_size(
               A;
               lambda_laplacian,
               method=step_method,
               iterations=power_iterations,
               safety,
           ) :
           float(step)
    _validate_positive(step, "step")

    return LaplacianFISTAWorkspace(
        A,
        b,
        _validate_image_size(image_size),
        x,
        copy(x),
        similar(x),
        zeros(eltype(x), length(b)),
        similar(x),
        similar(x),
        similar(x),
        similar(x),
        step,
    )
end

"""
    laplacian_lasso_step_size(A; lambda_laplacian=1.0, method=:exact, iterations=20, safety=1.05)

Return a conservative FISTA step size for the smooth part of
`0.5 * norm(A * x - b)^2 + 0.5 * lambda_laplacian * norm(L * x)^2`.

The estimate uses the bound `norm(L)^2 <= 64` for the 2D graph Laplacian.
"""
function laplacian_lasso_step_size(
    A;
    lambda_laplacian=1.0,
    method=:exact,
    iterations=20,
    safety=1.05,
)
    ndims(A) == 2 || throw(DimensionMismatch("A must be a matrix"))
    _validate_nonnegative(lambda_laplacian, "lambda_laplacian")
    _validate_positive(safety, "safety")

    data_norm2 = if method === :exact
        _least_squares_lipschitz(A)
    elseif method === :power
        iterations isa Integer && iterations > 0 ||
            throw(ArgumentError("iterations must be a positive integer"))
        _power_lipschitz(A, iterations)
    else
        throw(ArgumentError("method must be :exact or :power"))
    end
    lipschitz = data_norm2 + 64 * float(lambda_laplacian)
    return inv(float(safety) * (lipschitz > 0 ? lipschitz : one(float(lipschitz))))
end

function laplacian_lasso_fista(A, b, image_size, lambda_l1, lambda_laplacian; kwargs...)
    _validate_nonnegative(lambda_l1, "lambda_l1")
    _validate_nonnegative(lambda_laplacian, "lambda_laplacian")
    return _laplacian_lasso_fista(A, b, image_size, lambda_l1, lambda_laplacian; kwargs...)
end

function nonnegative_laplacian_lasso_fista(
    A,
    b,
    image_size,
    lambda_l1,
    lambda_laplacian;
    kwargs...,
)
    _validate_nonnegative(lambda_l1, "lambda_l1")
    _validate_nonnegative(lambda_laplacian, "lambda_laplacian")
    return _laplacian_lasso_fista(
        A,
        b,
        image_size,
        lambda_l1,
        lambda_laplacian;
        nonnegative=true,
        kwargs...,
    )
end

function laplacian_lasso_fista(solver::LaplacianFISTASolver, lambda_l1, lambda_laplacian; kwargs...)
    return laplacian_lasso_fista!(solver.workspace, lambda_l1, lambda_laplacian; kwargs...)
end

function nonnegative_laplacian_lasso_fista(
    solver::LaplacianFISTASolver,
    lambda_l1,
    lambda_laplacian;
    kwargs...,
)
    return nonnegative_laplacian_lasso_fista!(
        solver.workspace,
        lambda_l1,
        lambda_laplacian;
        kwargs...,
    )
end

function laplacian_lasso_fista!(
    workspace::LaplacianFISTAWorkspace,
    lambda_l1,
    lambda_laplacian;
    warm_start=false,
    kwargs...,
)
    _validate_nonnegative(lambda_l1, "lambda_l1")
    _validate_nonnegative(lambda_laplacian, "lambda_laplacian")
    return _laplacian_lasso_fista!(
        workspace,
        lambda_l1,
        lambda_laplacian;
        warm_start,
        kwargs...,
    )
end

function nonnegative_laplacian_lasso_fista!(
    workspace::LaplacianFISTAWorkspace,
    lambda_l1,
    lambda_laplacian;
    warm_start=false,
    kwargs...,
)
    _validate_nonnegative(lambda_l1, "lambda_l1")
    _validate_nonnegative(lambda_laplacian, "lambda_laplacian")
    return _laplacian_lasso_fista!(
        workspace,
        lambda_l1,
        lambda_laplacian;
        nonnegative=true,
        warm_start,
        kwargs...,
    )
end

function _laplacian_lasso_fista(
    A,
    b,
    image_size,
    lambda_l1,
    lambda_laplacian;
    step=nothing,
    x0=nothing,
    step_method=:exact,
    power_iterations=20,
    safety=1.05,
    kwargs...,
)
    if isnothing(step)
        step = laplacian_lasso_step_size(
            A;
            lambda_laplacian,
            method=step_method,
            iterations=power_iterations,
            safety,
        )
    end
    workspace = LaplacianFISTAWorkspace(A, b, image_size; step, x0)
    return _laplacian_lasso_fista!(
        workspace,
        lambda_l1,
        lambda_laplacian;
        warm_start=true,
        kwargs...,
    )
end

function _laplacian_lasso_fista!(
    workspace::LaplacianFISTAWorkspace,
    lambda_l1,
    lambda_laplacian;
    nonnegative=false,
    warm_start=false,
    abstol=1e-6,
    reltol=1e-4,
    maxiter=10_000,
    return_info=false,
)
    _validate_nonnegative(abstol, "abstol")
    _validate_nonnegative(reltol, "reltol")
    maxiter isa Integer && maxiter > 0 ||
        throw(ArgumentError("maxiter must be a positive integer"))

    x = workspace.x
    y = workspace.y
    x_previous = workspace.x_previous
    residual = workspace.residual
    gradient = workspace.gradient
    prox_input = workspace.prox_input
    laplacian = workspace.laplacian
    bilaplacian = workspace.bilaplacian
    step = workspace.step
    root_n = sqrt(length(x))

    if !warm_start
        fill!(x, zero(eltype(x)))
    end
    copyto!(y, x)

    iterate_change = Inf
    converged = false
    iterations = maxiter
    t = one(step)

    for iteration in 1:maxiter
        copyto!(x_previous, x)
        mul!(residual, workspace.A, y)
        @. residual = residual - workspace.b
        mul!(gradient, adjoint(workspace.A), residual)

        if !iszero(lambda_laplacian)
            _laplacian2d!(laplacian, y, workspace.image_size)
            _laplacian2d!(bilaplacian, laplacian, workspace.image_size)
            @. gradient = gradient + lambda_laplacian * bilaplacian
        end

        @. prox_input = y - step * gradient
        _prox_l1_nonnegative!(x, prox_input, step * lambda_l1, nonnegative)

        @. prox_input = x - x_previous
        iterate_change = norm(prox_input)
        iterate_tolerance =
            root_n * abstol + reltol * max(norm(x), norm(x_previous))
        if iterate_change <= iterate_tolerance
            converged = true
            iterations = iteration
            break
        end

        t_next = (one(t) + sqrt(one(t) + 4 * t^2)) / 2
        momentum = (t - one(t)) / t_next
        @. y = x + momentum * (x - x_previous)
        t = t_next
    end

    if return_info
        return (
            x=x,
            converged=converged,
            iterations=iterations,
            iterate_change=iterate_change,
            step=step,
            lambda_l1=lambda_l1,
            lambda_laplacian=lambda_laplacian,
            nonnegative=nonnegative,
        )
    end

    return x
end

function _laplacian2d!(result, x, image_size)
    height, width = image_size
    @inbounds for j in 1:width, i in 1:height
        k = i + (j - 1) * height
        center = x[k]
        value = zero(eltype(result))
        if i > 1
            value += x[k - 1] - center
        end
        if i < height
            value += x[k + 1] - center
        end
        if j > 1
            value += x[k - height] - center
        end
        if j < width
            value += x[k + height] - center
        end
        result[k] = value
    end
    return result
end
