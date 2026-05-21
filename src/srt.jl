module srt

using LinearAlgebra
using ProximalOperators

export LassoADMMSolver,
    LassoADMMWorkspace,
    LassoFISTASolver,
    LassoFISTAWorkspace,
    constrained_lasso_admm,
    constrained_lasso_admm!,
    constrained_lasso_fista,
    constrained_lasso_fista!,
    lasso_admm,
    lasso_admm!,
    lasso_fista,
    lasso_fista!,
    nonnegative_constrained_lasso_admm,
    nonnegative_constrained_lasso_admm!,
    nonnegative_constrained_lasso_fista,
    nonnegative_constrained_lasso_fista!,
    nonnegative_lasso_admm,
    nonnegative_lasso_admm!,
    nonnegative_lasso_fista,
    nonnegative_lasso_fista!

# Solve 0.5 * norm(A * x - b)^2 + lambda * norm(x, 1) with ADMM.
# Set return_info=true to also return termination details.
function lasso_admm(A, b, lambda; kwargs...)
    _validate_nonnegative(lambda, "lambda")
    return _lasso_admm(A, b, NormL1(lambda); kwargs...)
end


# Solve least squares over the L1 ball norm(x, 1) <= radius with ADMM.
# Set return_info=true to also return termination details.
function constrained_lasso_admm(A, b, radius; kwargs...)
    _validate_positive(radius, "radius")
    return _lasso_admm(A, b, IndBallL1(radius); kwargs...)
end

# Solve penalized LASSO with x >= 0. Zeros remain allowed for sparsity.
function nonnegative_lasso_admm(A, b, lambda; kwargs...)
    _validate_nonnegative(lambda, "lambda")
    return _lasso_admm(A, b, _NonnegativeNormL1(lambda); kwargs...)
end


# Solve least squares over the nonnegative L1 ball.
function nonnegative_constrained_lasso_admm(A, b, radius; kwargs...)
    _validate_positive(radius, "radius")
    return _lasso_admm(A, b, _NonnegativeBallL1(radius); kwargs...)
end

# Solve least squares over the L1 ball with FISTA.
function constrained_lasso_fista(A, b, radius; kwargs...)
    _validate_positive(radius, "radius")
    return _lasso_fista(A, b, IndBallL1(radius); kwargs...)
end

# Solve least squares over the nonnegative L1 ball with FISTA.
function nonnegative_constrained_lasso_fista(A, b, radius; kwargs...)
    _validate_positive(radius, "radius")
    return _lasso_fista(A, b, _NonnegativeBallL1(radius); kwargs...)
end

# Solve penalized LASSO with FISTA.
function lasso_fista(A, b, lambda; kwargs...)
    _validate_nonnegative(lambda, "lambda")
    return _lasso_fista(A, b, NormL1(lambda); kwargs...)
end

# Solve penalized LASSO with x >= 0 using FISTA.
function nonnegative_lasso_fista(A, b, lambda; kwargs...)
    _validate_nonnegative(lambda, "lambda")
    return _lasso_fista(A, b, _NonnegativeNormL1(lambda); kwargs...)
end

# Reuse this solver when A, b, and rho stay fixed across ADMM solves.
struct LassoADMMSolver{W}
    workspace::W
end

LassoADMMSolver(A, b; kwargs...) =
    LassoADMMSolver(LassoADMMWorkspace(A, b; kwargs...))

# Reuse this workspace when A, b, and rho stay fixed across LASSO solves.
mutable struct LassoADMMWorkspace{L,V,T}
    least_squares::L
    x::V
    z::V
    u::V
    z_previous::V
    prox_input::V
    residual::V
    rho::T
    gamma::T
end

function LassoADMMWorkspace(A, b; rho=1.0, x0=nothing)
    _validate_problem(A, b)
    _validate_positive(rho, "rho")
    rho = float(rho)

    z = _initial_iterate(A, b, x0)
    return LassoADMMWorkspace(
        LeastSquares(A, b),
        similar(z),
        z,
        zero(z),
        similar(z),
        similar(z),
        similar(z),
        rho,
        inv(rho),
    )
end

# Solver overloads reuse the cached ADMM state without exposing scratch buffers.
function lasso_admm(solver::LassoADMMSolver, lambda; kwargs...)
    return lasso_admm!(solver.workspace, lambda; kwargs...)
end

function constrained_lasso_admm(solver::LassoADMMSolver, radius; kwargs...)
    return constrained_lasso_admm!(solver.workspace, radius; kwargs...)
end

function nonnegative_lasso_admm(solver::LassoADMMSolver, lambda; kwargs...)
    return nonnegative_lasso_admm!(solver.workspace, lambda; kwargs...)
end

function nonnegative_constrained_lasso_admm(
    solver::LassoADMMSolver,
    radius;
    kwargs...,
)
    return nonnegative_constrained_lasso_admm!(solver.workspace, radius; kwargs...)
end

# Mutating solve that keeps the least-squares prox state cached in workspace.
# Pass warm_start=true to continue from workspace.z and workspace.u.
function lasso_admm!(
    workspace::LassoADMMWorkspace,
    lambda;
    warm_start=false,
    kwargs...,
)
    _validate_nonnegative(lambda, "lambda")
    return _lasso_admm!(workspace, NormL1(lambda); warm_start, kwargs...)
end

function constrained_lasso_admm!(
    workspace::LassoADMMWorkspace,
    radius;
    warm_start=false,
    kwargs...,
)
    _validate_positive(radius, "radius")
    return _lasso_admm!(workspace, IndBallL1(radius); warm_start, kwargs...)
end

function nonnegative_lasso_admm!(
    workspace::LassoADMMWorkspace,
    lambda;
    warm_start=false,
    kwargs...,
)
    _validate_nonnegative(lambda, "lambda")
    return _lasso_admm!(workspace, _NonnegativeNormL1(lambda); warm_start, kwargs...)
end

function nonnegative_constrained_lasso_admm!(
    workspace::LassoADMMWorkspace,
    radius;
    warm_start=false,
    kwargs...,
)
    _validate_positive(radius, "radius")
    return _lasso_admm!(workspace, _NonnegativeBallL1(radius); warm_start, kwargs...)
end

# Reuse this solver when A, b, and the FISTA step size stay fixed.
struct LassoFISTASolver{W}
    workspace::W
end

LassoFISTASolver(A, b; kwargs...) =
    LassoFISTASolver(LassoFISTAWorkspace(A, b; kwargs...))

mutable struct LassoFISTAWorkspace{M,B,V,R,T}
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

function LassoFISTAWorkspace(A, b; step=nothing, x0=nothing)
    _validate_problem(A, b)
    x = _initial_iterate(A, b, x0)
    step = isnothing(step) ? inv(_least_squares_lipschitz(A)) : float(step)
    _validate_positive(step, "step")
    residual = zeros(eltype(x), length(b))
    return LassoFISTAWorkspace(
        A,
        b,
        x,
        copy(x),
        similar(x),
        residual,
        similar(x),
        similar(x),
        step,
    )
end

function constrained_lasso_fista(solver::LassoFISTASolver, radius; kwargs...)
    return constrained_lasso_fista!(solver.workspace, radius; kwargs...)
end

function nonnegative_constrained_lasso_fista(
    solver::LassoFISTASolver,
    radius;
    kwargs...,
)
    return nonnegative_constrained_lasso_fista!(solver.workspace, radius; kwargs...)
end

function lasso_fista(solver::LassoFISTASolver, lambda; kwargs...)
    return lasso_fista!(solver.workspace, lambda; kwargs...)
end

function nonnegative_lasso_fista(solver::LassoFISTASolver, lambda; kwargs...)
    return nonnegative_lasso_fista!(solver.workspace, lambda; kwargs...)
end

function constrained_lasso_fista!(
    workspace::LassoFISTAWorkspace,
    radius;
    warm_start=false,
    kwargs...,
)
    _validate_positive(radius, "radius")
    return _lasso_fista!(workspace, IndBallL1(radius); warm_start, kwargs...)
end

function nonnegative_constrained_lasso_fista!(
    workspace::LassoFISTAWorkspace,
    radius;
    warm_start=false,
    kwargs...,
)
    _validate_positive(radius, "radius")
    return _lasso_fista!(workspace, _NonnegativeBallL1(radius); warm_start, kwargs...)
end

function lasso_fista!(
    workspace::LassoFISTAWorkspace,
    lambda;
    warm_start=false,
    kwargs...,
)
    _validate_nonnegative(lambda, "lambda")
    return _lasso_fista!(workspace, NormL1(lambda); warm_start, kwargs...)
end

function nonnegative_lasso_fista!(
    workspace::LassoFISTAWorkspace,
    lambda;
    warm_start=false,
    kwargs...,
)
    _validate_nonnegative(lambda, "lambda")
    return _lasso_fista!(workspace, _NonnegativeNormL1(lambda); warm_start, kwargs...)
end

struct _NonnegativeNormL1{T}
    l1::NormL1{T}
    nonnegative::IndNonnegative
end

_NonnegativeNormL1(lambda) = _NonnegativeNormL1(NormL1(lambda), IndNonnegative())

struct _NonnegativeBallL1{T}
    radius::T
    nonnegative::IndNonnegative
    simplex::IndSimplex{T}
end

_NonnegativeBallL1(radius::T) where {T} =
    _NonnegativeBallL1(radius, IndNonnegative(), IndSimplex(radius))

function _lasso_admm(
    A,
    b,
    nonsmooth_term;
    rho=1.0,
    x0=nothing,
    kwargs...,
)
    workspace = LassoADMMWorkspace(A, b; rho, x0)
    return _lasso_admm!(workspace, nonsmooth_term; warm_start=true, kwargs...)
end

function _lasso_admm!(
    workspace::LassoADMMWorkspace,
    nonsmooth_term;
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
    z = workspace.z
    u = workspace.u
    z_previous = workspace.z_previous
    prox_input = workspace.prox_input
    residual = workspace.residual
    rho = workspace.rho
    gamma = workspace.gamma
    root_n = sqrt(length(z))

    if !warm_start
        fill!(z, zero(eltype(z)))
        fill!(u, zero(eltype(u)))
    end

    primal_residual = Inf
    dual_residual = Inf
    converged = false
    iterations = maxiter

    for iteration in 1:maxiter
        @. prox_input = z - u
        prox!(x, workspace.least_squares, prox_input, gamma)
        copyto!(z_previous, z)
        @. prox_input = x + u
        _prox_nonsmooth!(z, nonsmooth_term, prox_input, gamma)

        @. residual = x - z
        @. u = u + residual

        primal_residual = norm(residual)
        @. residual = z - z_previous
        dual_residual = rho * norm(residual)
        primal_tolerance =
            root_n * abstol + reltol * max(norm(x), norm(z))
        dual_tolerance = root_n * abstol + reltol * rho * norm(u)

        if primal_residual <= primal_tolerance &&
           dual_residual <= dual_tolerance
            converged = true
            iterations = iteration
            break
        end
    end

    if return_info
        return (
            x=z,
            converged=converged,
            iterations=iterations,
            primal_residual=primal_residual,
            dual_residual=dual_residual,
        )
    end

    return z
end

function _lasso_fista(
    A,
    b,
    nonsmooth_term;
    step=nothing,
    x0=nothing,
    kwargs...,
)
    workspace = LassoFISTAWorkspace(A, b; step, x0)
    return _lasso_fista!(workspace, nonsmooth_term; warm_start=true, kwargs...)
end

function _lasso_fista!(
    workspace::LassoFISTAWorkspace,
    nonsmooth_term;
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
        @. prox_input = y - step * gradient
        _prox_nonsmooth!(x, nonsmooth_term, prox_input, step)

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
        )
    end

    return x
end

_prox_nonsmooth!(z, term, prox_input, gamma) = prox!(z, term, prox_input, gamma)

function _prox_nonsmooth!(z, term::_NonnegativeNormL1, prox_input, gamma)
    prox!(z, term.l1, prox_input, gamma)
    return prox!(z, term.nonnegative, z, gamma)
end

function _prox_nonsmooth!(z, term::_NonnegativeBallL1, prox_input, gamma)
    prox!(z, term.nonnegative, prox_input, gamma)
    if sum(z) > term.radius
        return prox!(z, term.simplex, prox_input, gamma)
    end
    return zero(eltype(z))
end

function _initial_iterate(A, b, x0)
    n = size(A, 2)
    T = float(promote_type(eltype(A), eltype(b)))

    if isnothing(x0)
        return zeros(T, n)
    end

    length(x0) == n ||
        throw(DimensionMismatch("x0 must have length $(n)"))
    return T.(x0)
end

function _least_squares_lipschitz(A)
    lipschitz = opnorm(A)^2
    return lipschitz > 0 ? lipschitz : one(float(lipschitz))
end

function _validate_problem(A, b)
    ndims(A) == 2 || throw(DimensionMismatch("A must be a matrix"))
    ndims(b) == 1 || throw(DimensionMismatch("b must be a vector"))
    size(A, 1) == length(b) ||
        throw(DimensionMismatch("A must have one row per entry of b"))
    return nothing
end

function _validate_positive(value, name)
    isfinite(value) && value > 0 ||
        throw(ArgumentError("$(name) must be finite and positive"))
    return nothing
end

function _validate_nonnegative(value, name)
    isfinite(value) && value >= 0 ||
        throw(ArgumentError("$(name) must be finite and nonnegative"))
    return nothing
end

end
