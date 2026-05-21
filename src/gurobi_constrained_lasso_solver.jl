using Gurobi
using JuMP

const GUROBI_ENV = Gurobi.Env(output_flag=0)

# Reuse the Gurobi environment across models to avoid repeated license setup.
gurobi_model() = Model(() -> Gurobi.Optimizer(GUROBI_ENV))

# Solve basis pursuit denoising: minimize norm(x, 1) with a residual bound.
function basis_pursuit_denoising_gurobi(
    A::AbstractMatrix,
    b::AbstractVector;
    noise=0.0,
    positive=false,
    verbose=false,
)
    _validate_problem(A, b)
    _validate_nonnegative(noise, "noise")

    model, x = _gurobi_l1_model(A; positive, verbose)
    @variable(model, l1_norm)
    @constraint(model, [l1_norm; x] in MOI.NormOneCone(1 + length(x)))

    if noise > 0
        @constraint(model, [noise; A * x - b] in MOI.SecondOrderCone(1 + length(b)))
    else
        @constraint(model, A * x == b)
    end

    @objective(model, Min, l1_norm)
    return _solve_gurobi(model, x)
end

# Solve constrained LASSO: minimize least squares over an L1 ball.
function constrained_lasso_gurobi(
    A::AbstractMatrix,
    b::AbstractVector,
    radius;
    positive=false,
    verbose=false,
)
    _validate_problem(A, b)
    _validate_positive(radius, "radius")

    model, x = _gurobi_l1_model(A; positive, verbose)
    @constraint(model, [radius; x] in MOI.NormOneCone(1 + length(x)))
    residual = _gurobi_residual(model, A, b, x)
    @objective(model, Min, 0.5 * sum(residual .^ 2))
    return _solve_gurobi(model, x)
end

# Solve penalized LASSO: minimize least squares plus an L1 penalty.
function lasso_gurobi(
    A::AbstractMatrix,
    b::AbstractVector,
    lambda;
    positive=false,
    verbose=false,
)
    _validate_problem(A, b)
    _validate_nonnegative(lambda, "lambda")

    model, x = _gurobi_l1_model(A; positive, verbose)
    @variable(model, l1_norm)
    @constraint(model, [l1_norm; x] in MOI.NormOneCone(1 + length(x)))
    residual = _gurobi_residual(model, A, b, x)
    @objective(model, Min, 0.5 * sum(residual .^ 2) + lambda * l1_norm)
    return _solve_gurobi(model, x)
end

# Retain the original equality-constrained helper under a truthful name.
# The residual objective term is zero whenever the model is feasible.
function equality_constrained_lasso_gurobi(
    A::AbstractMatrix,
    b::AbstractVector,
    lambda;
    positive=false,
    verbose=false,
)
    _validate_problem(A, b)
    _validate_nonnegative(lambda, "lambda")

    model, x = _gurobi_l1_model(A; positive, verbose)
    @variable(model, l1_norm)
    @constraint(model, [l1_norm; x] in MOI.NormOneCone(1 + length(x)))
    @constraint(model, A * x == b)
    @objective(model, Min, 0.5 * sum((A * x - b) .^ 2) + lambda * l1_norm)
    return _solve_gurobi(model, x)
end

function _gurobi_l1_model(A; positive=false, verbose=false)
    model = gurobi_model()
    !verbose && set_attribute(model, "LogToConsole", 0)
    @variable(model, x[1:size(A, 2)])
    positive && @constraint(model, x .>= 0)
    return model, x
end

function _gurobi_residual(model, A, b, x)
    @variable(model, residual[1:length(b)])
    @constraint(model, residual .== A * x - b)
    return residual
end

function _solve_gurobi(model, x)
    optimize!(model)
    is_solved_and_feasible(model) ||
        throw(ErrorException("Gurobi did not return a feasible solution"))
    return value.(x)
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
