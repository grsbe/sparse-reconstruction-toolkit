using LinearAlgebra
using ProximalOperators

export LassoADMMSolver,
    LassoADMMWorkspace,
    LassoFISTASolver,
    LassoFISTAWorkspace,
    LassoRidgeFISTASolver,
    LassoRidgeFISTAWorkspace,
    constrained_lasso_admm,
    constrained_lasso_admm!,
    constrained_lasso_fista,
    constrained_lasso_fista!,
    fista_step_size,
    lasso_admm,
    lasso_admm!,
    lasso_fista,
    lasso_fista!,
    lasso_ridge_fista,
    lasso_ridge_fista!,
    nonnegative_lasso_ridge_fista,
    nonnegative_lasso_ridge_fista!,
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

# Solve 0.5 * norm(A * x + B * z - b)^2 with L1 penalty on x and ridge on z.
function lasso_ridge_fista(A, B, b, lambda_x, lambda_z; kwargs...)
    _validate_nonnegative(lambda_x, "lambda_x")
    _validate_nonnegative(lambda_z, "lambda_z")
    return _lasso_ridge_fista(A, B, b, lambda_x, lambda_z; kwargs...)
end

# Solve the LASSO-ridge problem with x >= 0 and z >= 0 using FISTA.
function nonnegative_lasso_ridge_fista(A, B, b, lambda_x, lambda_z; kwargs...)
    _validate_nonnegative(lambda_x, "lambda_x")
    _validate_nonnegative(lambda_z, "lambda_z")
    return _lasso_ridge_fista(
        A,
        B,
        b,
        lambda_x,
        lambda_z;
        x_nonnegative=true,
        z_nonnegative=true,
        kwargs...,
    )
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
    step = isnothing(step) ? fista_step_size(A) : float(step)
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

# Reuse this solver when A, B, b, and the FISTA step size stay fixed.
struct LassoRidgeFISTASolver{W}
    workspace::W
end

LassoRidgeFISTASolver(A, B, b; kwargs...) =
    LassoRidgeFISTASolver(LassoRidgeFISTAWorkspace(A, B, b; kwargs...))

mutable struct LassoRidgeFISTAWorkspace{MA,MB,B,VX,VZ,R,T}
    A::MA
    B::MB
    b::B
    x::VX
    z::VZ
    x_momentum::VX
    z_momentum::VZ
    x_previous::VX
    z_previous::VZ
    residual::R
    gradient_x::VX
    gradient_z::VZ
    prox_input_x::VX
    prox_input_z::VZ
    step::T
end

function LassoRidgeFISTAWorkspace(
    A,
    B,
    b;
    step=nothing,
    x0=nothing,
    z0=nothing,
)
    _validate_block_problem(A, B, b)
    x = _initial_iterate(A, b, x0)
    z = _initial_iterate(B, b, z0)
    step = isnothing(step) ? fista_step_size(A, B) : float(step)
    _validate_positive(step, "step")
    residual_type = promote_type(eltype(x), eltype(z))
    residual = zeros(residual_type, length(b))
    return LassoRidgeFISTAWorkspace(
        A,
        B,
        b,
        x,
        z,
        copy(x),
        copy(z),
        similar(x),
        similar(z),
        residual,
        similar(x),
        similar(z),
        similar(x),
        similar(z),
        step,
    )
end

"""
    fista_step_size(A; method=:exact, iterations=20, safety=1.05)

Return a FISTA step size for the least-squares term `0.5 * norm(A * x - b)^2`.
`method=:exact` uses `opnorm(A)`; `method=:power` uses power iterations and a
small safety factor to avoid a costly exact spectral norm on large operators.
"""
function fista_step_size(A; method=:exact, iterations=20, safety=1.05)
    ndims(A) == 2 || throw(DimensionMismatch("A must be a matrix"))

    if method === :exact
        return inv(_least_squares_lipschitz(A))
    elseif method === :power
        iterations isa Integer && iterations > 0 ||
            throw(ArgumentError("iterations must be a positive integer"))
        _validate_positive(safety, "safety")
        return inv(float(safety) * _power_lipschitz(A, iterations))
    end

    throw(ArgumentError("method must be :exact or :power"))
end

"""
    fista_step_size(A, B; method=:exact, iterations=20, safety=1.05)

Return a FISTA step size for the two-block least-squares term
`0.5 * norm(A * x + B * z - b)^2`.
"""
function fista_step_size(A, B; method=:exact, iterations=20, safety=1.05)
    _validate_block_design(A, B)

    if method === :exact
        return inv(_block_least_squares_lipschitz(A, B))
    elseif method === :power
        iterations isa Integer && iterations > 0 ||
            throw(ArgumentError("iterations must be a positive integer"))
        _validate_positive(safety, "safety")
        return inv(float(safety) * _block_power_lipschitz(A, B, iterations))
    end

    throw(ArgumentError("method must be :exact or :power"))
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

function lasso_ridge_fista(
    solver::LassoRidgeFISTASolver,
    lambda_x,
    lambda_z;
    kwargs...,
)
    return lasso_ridge_fista!(solver.workspace, lambda_x, lambda_z; kwargs...)
end

function nonnegative_lasso_ridge_fista(
    solver::LassoRidgeFISTASolver,
    lambda_x,
    lambda_z;
    kwargs...,
)
    return nonnegative_lasso_ridge_fista!(solver.workspace, lambda_x, lambda_z; kwargs...)
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

function lasso_ridge_fista!(
    workspace::LassoRidgeFISTAWorkspace,
    lambda_x,
    lambda_z;
    warm_start=false,
    kwargs...,
)
    _validate_nonnegative(lambda_x, "lambda_x")
    _validate_nonnegative(lambda_z, "lambda_z")
    return _lasso_ridge_fista!(
        workspace,
        lambda_x,
        lambda_z;
        warm_start,
        kwargs...,
    )
end

function nonnegative_lasso_ridge_fista!(
    workspace::LassoRidgeFISTAWorkspace,
    lambda_x,
    lambda_z;
    warm_start=false,
    kwargs...,
)
    _validate_nonnegative(lambda_x, "lambda_x")
    _validate_nonnegative(lambda_z, "lambda_z")
    return _lasso_ridge_fista!(
        workspace,
        lambda_x,
        lambda_z;
        x_nonnegative=true,
        z_nonnegative=true,
        warm_start,
        kwargs...,
    )
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
    optimality_abstol=nothing,
    optimality_reltol=nothing,
    maxiter=10_000,
    return_info=false,
)
    _validate_nonnegative(abstol, "abstol")
    _validate_nonnegative(reltol, "reltol")
    isnothing(optimality_abstol) ||
        _validate_nonnegative(optimality_abstol, "optimality_abstol")
    isnothing(optimality_reltol) ||
        _validate_nonnegative(optimality_reltol, "optimality_reltol")
    maxiter isa Integer && maxiter > 0 ||
        throw(ArgumentError("maxiter must be a positive integer"))
    use_optimality = !isnothing(optimality_abstol) || !isnothing(optimality_reltol)
    optimality_abstol = something(optimality_abstol, zero(workspace.step))
    optimality_reltol = something(optimality_reltol, zero(workspace.step))

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
    optimality_residual = Inf
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
        @. prox_input = (y - x) / step
        optimality_residual = norm(prox_input, Inf)
        optimality_scale = max(norm(gradient, Inf), eps(float(step)))
        iterate_tolerance = root_n * abstol + reltol * max(norm(x), norm(x_previous))
        optimality_tolerance =
            optimality_abstol + optimality_reltol * optimality_scale
        if use_optimality ?
           optimality_residual <= optimality_tolerance :
           iterate_change <= iterate_tolerance
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
            optimality_residual=optimality_residual,
            step=step,
        )
    end

    return x
end

function _lasso_ridge_fista(
    A,
    B,
    b,
    lambda_x,
    lambda_z;
    step=nothing,
    x0=nothing,
    z0=nothing,
    kwargs...,
)
    workspace = LassoRidgeFISTAWorkspace(A, B, b; step, x0, z0)
    return _lasso_ridge_fista!(
        workspace,
        lambda_x,
        lambda_z;
        warm_start=true,
        kwargs...,
    )
end

function _lasso_ridge_fista!(
    workspace::LassoRidgeFISTAWorkspace,
    lambda_x,
    lambda_z;
    x_nonnegative=false,
    z_nonnegative=false,
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
    x_momentum = workspace.x_momentum
    z_momentum = workspace.z_momentum
    x_previous = workspace.x_previous
    z_previous = workspace.z_previous
    residual = workspace.residual
    gradient_x = workspace.gradient_x
    gradient_z = workspace.gradient_z
    prox_input_x = workspace.prox_input_x
    prox_input_z = workspace.prox_input_z
    step = workspace.step
    x_term = x_nonnegative ? _NonnegativeNormL1(lambda_x) : NormL1(lambda_x)
    ridge_scale = inv(one(step) + step * lambda_z)
    root_n = sqrt(length(x) + length(z))

    if !warm_start
        fill!(x, zero(eltype(x)))
        fill!(z, zero(eltype(z)))
    end
    copyto!(x_momentum, x)
    copyto!(z_momentum, z)

    iterate_change = Inf
    converged = false
    iterations = maxiter
    t = one(step)

    for iteration in 1:maxiter
        copyto!(x_previous, x)
        copyto!(z_previous, z)
        mul!(residual, workspace.A, x_momentum)
        mul!(residual, workspace.B, z_momentum, one(step), one(step))
        @. residual = residual - workspace.b
        mul!(gradient_x, adjoint(workspace.A), residual)
        mul!(gradient_z, adjoint(workspace.B), residual)
        @. prox_input_x = x_momentum - step * gradient_x
        @. prox_input_z = z_momentum - step * gradient_z
        _prox_nonsmooth!(x, x_term, prox_input_x, step)
        _prox_ridge!(z, prox_input_z, ridge_scale, z_nonnegative)

        @. prox_input_x = x - x_previous
        @. prox_input_z = z - z_previous
        iterate_change = sqrt(norm(prox_input_x)^2 + norm(prox_input_z)^2)
        iterate_norm = max(
            sqrt(norm(x)^2 + norm(z)^2),
            sqrt(norm(x_previous)^2 + norm(z_previous)^2),
        )
        iterate_tolerance = root_n * abstol + reltol * iterate_norm
        if iterate_change <= iterate_tolerance
            converged = true
            iterations = iteration
            break
        end

        t_next = (one(t) + sqrt(one(t) + 4 * t^2)) / 2
        momentum = (t - one(t)) / t_next
        @. x_momentum = x + momentum * (x - x_previous)
        @. z_momentum = z + momentum * (z - z_previous)
        t = t_next
    end

    if return_info
        return (
            x=x,
            z=z,
            converged=converged,
            iterations=iterations,
            iterate_change=iterate_change,
            step=step,
        )
    end

    return (x=x, z=z)
end

_prox_nonsmooth!(z, term, prox_input, gamma) = prox!(z, term, prox_input, gamma)

function _prox_ridge!(z, prox_input, ridge_scale, nonnegative)
    if nonnegative
        @. z = ridge_scale * max(prox_input, zero(eltype(z)))
    else
        @. z = ridge_scale * prox_input
    end
    return z
end

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

function _block_least_squares_lipschitz(A, B)
    lipschitz = opnorm(hcat(A, B))^2
    return lipschitz > 0 ? lipschitz : one(float(lipschitz))
end

function _power_lipschitz(A, iterations)
    T = float(eltype(A))
    x = fill(inv(sqrt(T(size(A, 2)))), size(A, 2))
    Ax = zeros(T, size(A, 1))
    AtAx = similar(x)

    for _ in 1:iterations
        mul!(Ax, A, x)
        mul!(AtAx, adjoint(A), Ax)
        AtAx_norm = norm(AtAx)
        if iszero(AtAx_norm)
            return one(T)
        end
        @. x = AtAx / AtAx_norm
    end

    mul!(Ax, A, x)
    lipschitz = norm(Ax)^2
    return lipschitz > 0 ? lipschitz : one(T)
end

function _block_power_lipschitz(A, B, iterations)
    T = float(promote_type(eltype(A), eltype(B)))
    nblocks = size(A, 2) + size(B, 2)
    scale = inv(sqrt(T(nblocks)))
    x = fill(scale, size(A, 2))
    z = fill(scale, size(B, 2))
    residual = zeros(T, size(A, 1))
    gradient_x = similar(x)
    gradient_z = similar(z)

    for _ in 1:iterations
        mul!(residual, A, x)
        mul!(residual, B, z, one(T), one(T))
        mul!(gradient_x, adjoint(A), residual)
        mul!(gradient_z, adjoint(B), residual)
        gradient_norm = sqrt(norm(gradient_x)^2 + norm(gradient_z)^2)
        if iszero(gradient_norm)
            return one(T)
        end
        @. x = gradient_x / gradient_norm
        @. z = gradient_z / gradient_norm
    end

    mul!(residual, A, x)
    mul!(residual, B, z, one(T), one(T))
    lipschitz = norm(residual)^2
    return lipschitz > 0 ? lipschitz : one(T)
end

function _validate_problem(A, b)
    ndims(A) == 2 || throw(DimensionMismatch("A must be a matrix"))
    ndims(b) == 1 || throw(DimensionMismatch("b must be a vector"))
    size(A, 1) == length(b) ||
        throw(DimensionMismatch("A must have one row per entry of b"))
    return nothing
end

function _validate_block_design(A, B)
    ndims(A) == 2 || throw(DimensionMismatch("A must be a matrix"))
    ndims(B) == 2 || throw(DimensionMismatch("B must be a matrix"))
    size(A, 1) == size(B, 1) ||
        throw(DimensionMismatch("A and B must have the same number of rows"))
    return nothing
end

function _validate_block_problem(A, B, b)
    _validate_block_design(A, B)
    ndims(b) == 1 || throw(DimensionMismatch("b must be a vector"))
    size(A, 1) == length(b) ||
        throw(DimensionMismatch("A and B must have one row per entry of b"))
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
