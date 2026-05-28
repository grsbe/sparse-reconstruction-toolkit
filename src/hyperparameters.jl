using Random

export lasso_crossvalidation,
    lasso_noise_sweep,
    nonnegative_lasso_crossvalidation,
    nonnegative_lasso_noise_sweep

"""
    lasso_noise_sweep(A, b, noise; algorithm=:fista, steps=10, kwargs...)

Choose the penalized LASSO hyperparameter `lambda` by a warm-started bisection
search whose target is `norm(A * x - b) approximately noise`.

Set `algorithm` to `:fista` or `:admm`. Extra keyword arguments are passed to
the inner solver, except for `step`, `rho`, and `x0`, which configure the
reused solver workspace. The result is a named tuple containing the selected
`lambda`, the selected solver `info`, the residual norm, residual gap, final
bracket, and per-trial history.
"""
function lasso_noise_sweep(
    A,
    b,
    noise;
    algorithm=:fista,
    steps=10,
    lambda_high=nothing,
    x0=nothing,
    step=nothing,
    rho=1.0,
    noise_abstol=0.0,
    noise_reltol=0.0,
    kwargs...,
)
    return _lasso_noise_sweep(
        A,
        b,
        noise,
        false;
        algorithm,
        steps,
        lambda_high,
        x0,
        step,
        rho,
        noise_abstol,
        noise_reltol,
        kwargs...,
    )
end

"""
    nonnegative_lasso_noise_sweep(A, b, noise; algorithm=:fista, steps=10, kwargs...)

Choose `lambda` for nonnegative penalized LASSO by matching
`norm(A * x - b)` to a supplied L2 noise norm with a warm-started bisection
search.
"""
function nonnegative_lasso_noise_sweep(
    A,
    b,
    noise;
    algorithm=:fista,
    steps=10,
    lambda_high=nothing,
    x0=nothing,
    step=nothing,
    rho=1.0,
    noise_abstol=0.0,
    noise_reltol=0.0,
    kwargs...,
)
    return _lasso_noise_sweep(
        A,
        b,
        noise,
        true;
        algorithm,
        steps,
        lambda_high,
        x0,
        step,
        rho,
        noise_abstol,
        noise_reltol,
        kwargs...,
    )
end

function _lasso_noise_sweep(
    A,
    b,
    noise,
    nonnegative;
    algorithm,
    steps,
    lambda_high,
    x0,
    step,
    rho,
    noise_abstol,
    noise_reltol,
    kwargs...,
)
    _validate_problem(A, b)
    _validate_nonnegative(noise, "noise")
    steps isa Integer && steps > 0 ||
        throw(ArgumentError("steps must be a positive integer"))
    _validate_nonnegative(noise_abstol, "noise_abstol")
    _validate_nonnegative(noise_reltol, "noise_reltol")
    _validate_noise_sweep_kwargs(kwargs)

    lambda_high = isnothing(lambda_high) ?
                  _lasso_lambda_max(A, b, nonnegative) :
                  float(lambda_high)
    _validate_positive(lambda_high, "lambda_high")

    solver = _noise_sweep_solver(A, b, algorithm; x0, step, rho)
    lambda_low = zero(lambda_high)
    best = nothing
    trials = NamedTuple[]
    tolerance = noise_abstol + noise_reltol * max(noise, eps(float(noise)))

    for trial in 1:steps
        lambda = trial == 1 ? lambda_high / 2 : (lambda_low + lambda_high) / 2
        info = _noise_sweep_solve(
            solver,
            lambda,
            algorithm,
            nonnegative;
            warm_start=trial > 1,
            kwargs...,
        )
        info = merge(info, (x=copy(info.x),))
        residual = norm(A * info.x - b)
        gap = residual - noise
        candidate = (
            trial=trial,
            lambda=lambda,
            info=info,
            residual=residual,
            gap=gap,
        )
        push!(trials, candidate)

        if isnothing(best) || abs(gap) < abs(best.gap)
            best = candidate
        end

        if gap > 0
            lambda_high = lambda
        else
            lambda_low = lambda
        end

        if abs(gap) <= tolerance
            break
        end
    end

    return (
        lambda=best.lambda,
        info=best.info,
        x=best.info.x,
        residual=best.residual,
        gap=best.gap,
        lambda_low=lambda_low,
        lambda_high=lambda_high,
        trials=trials,
    )
end

"""
    lasso_crossvalidation(A, b; folds=5, lambdas=nothing, algorithm=:fista, kwargs...)

Choose the penalized LASSO hyperparameter `lambda` by k-fold cross-validation
over rows of `A` and entries of `b`. For each candidate lambda, the solver is
fit on `folds - 1` parts and scored by mean squared residual on the held-out
part. The selected lambda is refit on all data before returning.

By default, a descending geometric lambda grid is built from `norm(A' * b, Inf)`.
Pass `folds=3` or `folds=5` for common 3-way or 5-way CV, and set
`shuffle=true` to randomly permute rows before splitting.
"""
function lasso_crossvalidation(
    A,
    b;
    folds=5,
    lambdas=nothing,
    algorithm=:fista,
    grid_length=20,
    min_ratio=1e-3,
    shuffle=false,
    rng=Random.default_rng(),
    x0=nothing,
    step=nothing,
    rho=1.0,
    kwargs...,
)
    return _lasso_crossvalidation(
        A,
        b,
        false;
        folds,
        lambdas,
        algorithm,
        grid_length,
        min_ratio,
        shuffle,
        rng,
        x0,
        step,
        rho,
        kwargs...,
    )
end

"""
    nonnegative_lasso_crossvalidation(A, b; folds=5, lambdas=nothing, algorithm=:fista, kwargs...)

Choose `lambda` for nonnegative penalized LASSO by k-fold cross-validation over
rows of the data matrix.
"""
function nonnegative_lasso_crossvalidation(
    A,
    b;
    folds=5,
    lambdas=nothing,
    algorithm=:fista,
    grid_length=20,
    min_ratio=1e-3,
    shuffle=false,
    rng=Random.default_rng(),
    x0=nothing,
    step=nothing,
    rho=1.0,
    kwargs...,
)
    return _lasso_crossvalidation(
        A,
        b,
        true;
        folds,
        lambdas,
        algorithm,
        grid_length,
        min_ratio,
        shuffle,
        rng,
        x0,
        step,
        rho,
        kwargs...,
    )
end

function _lasso_crossvalidation(
    A,
    b,
    nonnegative;
    folds,
    lambdas,
    algorithm,
    grid_length,
    min_ratio,
    shuffle,
    rng,
    x0,
    step,
    rho,
    kwargs...,
)
    _validate_problem(A, b)
    _validate_crossvalidation_kwargs(kwargs)
    folds isa Integer && 2 <= folds <= length(b) ||
        throw(ArgumentError("folds must be an integer between 2 and length(b)"))
    lambda_grid = _crossvalidation_lambdas(
        A,
        b,
        nonnegative,
        lambdas;
        grid_length,
        min_ratio,
    )

    row_folds = _crossvalidation_folds(length(b), folds; shuffle, rng)
    fold_errors = zeros(float(promote_type(eltype(A), eltype(b))), length(lambda_grid), folds)

    for fold in 1:folds
        validation = row_folds[fold]
        training = _training_indices(length(b), validation)
        A_train = A[training, :]
        b_train = b[training]
        A_validation = A[validation, :]
        b_validation = b[validation]
        solver = _noise_sweep_solver(A_train, b_train, algorithm; x0, step, rho)

        for (index, lambda) in pairs(lambda_grid)
            info = _noise_sweep_solve(
                solver,
                lambda,
                algorithm,
                nonnegative;
                warm_start=index > 1,
                kwargs...,
            )
            residual = A_validation * info.x - b_validation
            fold_errors[index, fold] = sum(abs2, residual) / length(b_validation)
        end
    end

    fold_sizes = length.(row_folds)
    validation_errors = fold_errors * fold_sizes ./ length(b)
    best_index = argmin(validation_errors)
    final_solver = _noise_sweep_solver(A, b, algorithm; x0, step, rho)
    info = _noise_sweep_solve(
        final_solver,
        lambda_grid[best_index],
        algorithm,
        nonnegative;
        warm_start=false,
        kwargs...,
    )
    info = merge(info, (x=copy(info.x),))

    return (
        lambda=lambda_grid[best_index],
        info=info,
        x=info.x,
        validation_error=validation_errors[best_index],
        validation_errors=validation_errors,
        fold_errors=fold_errors,
        fold_sizes=fold_sizes,
        lambdas=lambda_grid,
        fold_indices=row_folds,
    )
end

function _validate_crossvalidation_kwargs(kwargs)
    haskey(kwargs, :warm_start) &&
        throw(ArgumentError("warm_start is managed by cross-validation"))
    haskey(kwargs, :return_info) &&
        throw(ArgumentError("return_info is managed by cross-validation"))
    return nothing
end

function _crossvalidation_lambdas(
    A,
    b,
    nonnegative,
    lambdas;
    grid_length,
    min_ratio,
)
    if !isnothing(lambdas)
        lambda_grid = float.(collect(lambdas))
        !isempty(lambda_grid) || throw(ArgumentError("lambdas must be nonempty"))
        for lambda in lambda_grid
            _validate_nonnegative(lambda, "lambda")
        end
        return lambda_grid
    end

    grid_length isa Integer && grid_length > 0 ||
        throw(ArgumentError("grid_length must be a positive integer"))
    _validate_positive(min_ratio, "min_ratio")
    min_ratio <= 1 || throw(ArgumentError("min_ratio must be <= 1"))
    lambda_max = _lasso_lambda_max(A, b, nonnegative)
    if grid_length == 1
        return [lambda_max]
    end
    scale = exp.(range(0, log(min_ratio); length=grid_length))
    return lambda_max .* scale
end

function _crossvalidation_folds(n, folds; shuffle, rng)
    indices = collect(1:n)
    if shuffle
        Random.shuffle!(rng, indices)
    end
    row_folds = Vector{Vector{Int}}(undef, folds)
    start = 1
    base_size = div(n, folds)
    extra = rem(n, folds)
    for fold in 1:folds
        fold_size = base_size + (fold <= extra ? 1 : 0)
        stop = start + fold_size - 1
        row_folds[fold] = indices[start:stop]
        start = stop + 1
    end
    return row_folds
end

function _training_indices(n, validation)
    is_training = trues(n)
    is_training[validation] .= false
    return findall(is_training)
end

function _validate_noise_sweep_kwargs(kwargs)
    haskey(kwargs, :warm_start) &&
        throw(ArgumentError("warm_start is managed by the noise sweep"))
    haskey(kwargs, :return_info) &&
        throw(ArgumentError("return_info is managed by the noise sweep"))
    return nothing
end

function _noise_sweep_solver(A, b, algorithm; x0, step, rho)
    if algorithm === :fista
        return isnothing(step) ?
               LassoFISTASolver(A, b; x0) :
               LassoFISTASolver(A, b; x0, step)
    elseif algorithm === :admm
        return LassoADMMSolver(A, b; x0, rho)
    end

    throw(ArgumentError("algorithm must be :fista or :admm"))
end

function _noise_sweep_solve(
    solver,
    lambda,
    algorithm,
    nonnegative;
    warm_start,
    kwargs...,
)
    if algorithm === :fista
        return nonnegative ?
               nonnegative_lasso_fista(
                   solver,
                   lambda;
                   warm_start,
                   return_info=true,
                   kwargs...,
               ) :
               lasso_fista(
                   solver,
                   lambda;
                   warm_start,
                   return_info=true,
                   kwargs...,
               )
    end

    return nonnegative ?
           nonnegative_lasso_admm(
               solver,
               lambda;
               warm_start,
               return_info=true,
               kwargs...,
           ) :
           lasso_admm(
               solver,
               lambda;
               warm_start,
               return_info=true,
               kwargs...,
           )
end

function _lasso_lambda_max(A, b, nonnegative)
    gradient = adjoint(A) * b
    lambda_max = nonnegative ? maximum(gradient) : norm(gradient, Inf)
    return max(float(lambda_max), eps(float(lambda_max)))
end
