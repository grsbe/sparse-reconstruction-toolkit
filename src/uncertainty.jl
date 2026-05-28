export debiased_lasso_refit,
    lasso_bootstrap_uncertainty,
    lasso_lambda_path,
    lasso_noise_perturbation_uncertainty,
    lasso_stability_selection,
    nonnegative_lasso_bootstrap_uncertainty,
    nonnegative_lasso_lambda_path,
    nonnegative_lasso_noise_perturbation_uncertainty,
    nonnegative_lasso_stability_selection

"""
    lasso_bootstrap_uncertainty(A, b, lambda; samples=100, kwargs...)

Estimate coefficient uncertainty by resampling rows of `(A, b)` with replacement
and resolving LASSO. Returns coefficient draws, means, standard deviations,
quantile intervals, selection probabilities, and prediction summaries on the
original rows.
"""
function lasso_bootstrap_uncertainty(A, b, lambda; kwargs...)
    return _lasso_resampling_uncertainty(A, b, lambda, false, :bootstrap; kwargs...)
end

"""
    nonnegative_lasso_bootstrap_uncertainty(A, b, lambda; samples=100, kwargs...)

Bootstrap uncertainty for nonnegative penalized LASSO.
"""
function nonnegative_lasso_bootstrap_uncertainty(A, b, lambda; kwargs...)
    return _lasso_resampling_uncertainty(A, b, lambda, true, :bootstrap; kwargs...)
end

"""
    lasso_noise_perturbation_uncertainty(A, b, lambda; noise_sigma, samples=100, kwargs...)

Estimate sensitivity to an assumed Gaussian measurement-noise model by repeatedly
solving with `b + noise_sigma * randn(...)`. Alternatively pass `noise_norm`,
which is converted to `noise_sigma = noise_norm / sqrt(length(b))`.
"""
function lasso_noise_perturbation_uncertainty(A, b, lambda; kwargs...)
    return _lasso_resampling_uncertainty(A, b, lambda, false, :noise; kwargs...)
end

"""
    nonnegative_lasso_noise_perturbation_uncertainty(A, b, lambda; noise_sigma, samples=100, kwargs...)

Noise-perturbation uncertainty for nonnegative penalized LASSO.
"""
function nonnegative_lasso_noise_perturbation_uncertainty(A, b, lambda; kwargs...)
    return _lasso_resampling_uncertainty(A, b, lambda, true, :noise; kwargs...)
end

function _lasso_resampling_uncertainty(
    A,
    b,
    lambda,
    nonnegative,
    method;
    samples=100,
    algorithm=:fista,
    rng=Random.default_rng(),
    confidence_level=0.95,
    selection_tol=1e-8,
    noise_sigma=nothing,
    noise_norm=nothing,
    x0=nothing,
    step=nothing,
    rho=1.0,
    kwargs...,
)
    _validate_problem(A, b)
    _validate_nonnegative(lambda, "lambda")
    samples isa Integer && samples > 0 ||
        throw(ArgumentError("samples must be a positive integer"))
    _validate_confidence_level(confidence_level)
    _validate_nonnegative(selection_tol, "selection_tol")
    _validate_uncertainty_kwargs(kwargs)
    sigma = _noise_sigma(noise_sigma, noise_norm, length(b), method)

    nfeatures = size(A, 2)
    coefficient_draws = zeros(float(promote_type(eltype(A), eltype(b))), nfeatures, samples)

    for sample in 1:samples
        if method === :bootstrap
            rows = rand(rng, 1:length(b), length(b))
            sample_A = A[rows, :]
            sample_b = b[rows]
        else
            sample_A = A
            sample_b = b .+ sigma .* randn(rng, length(b))
        end
        info = _solve_lasso_sample(
            sample_A,
            sample_b,
            lambda,
            algorithm,
            nonnegative;
            x0,
            step,
            rho,
            kwargs...,
        )
        coefficient_draws[:, sample] = info.x
    end

    return _uncertainty_summary(A, coefficient_draws; confidence_level, selection_tol)
end

"""
    lasso_stability_selection(A, b, lambda; samples=100, subsample_fraction=0.5, kwargs...)

Repeatedly solve LASSO on random row subsets and return per-coefficient selection
probabilities. Pass a vector of lambdas to compare support stability across a
lambda grid.
"""
function lasso_stability_selection(A, b, lambda; kwargs...)
    return _lasso_stability_selection(A, b, lambda, false; kwargs...)
end

"""
    nonnegative_lasso_stability_selection(A, b, lambda; samples=100, subsample_fraction=0.5, kwargs...)

Stability selection for nonnegative penalized LASSO.
"""
function nonnegative_lasso_stability_selection(A, b, lambda; kwargs...)
    return _lasso_stability_selection(A, b, lambda, true; kwargs...)
end

function _lasso_stability_selection(
    A,
    b,
    lambda,
    nonnegative;
    samples=100,
    subsample_fraction=0.5,
    threshold=0.8,
    selection_tol=1e-8,
    algorithm=:fista,
    rng=Random.default_rng(),
    x0=nothing,
    step=nothing,
    rho=1.0,
    kwargs...,
)
    _validate_problem(A, b)
    samples isa Integer && samples > 0 ||
        throw(ArgumentError("samples must be a positive integer"))
    _validate_positive(subsample_fraction, "subsample_fraction")
    subsample_fraction <= 1 || throw(ArgumentError("subsample_fraction must be <= 1"))
    _validate_nonnegative(selection_tol, "selection_tol")
    _validate_probability(threshold, "threshold")
    _validate_uncertainty_kwargs(kwargs)

    lambdas = float.(collect(lambda isa Number ? [lambda] : lambda))
    !isempty(lambdas) || throw(ArgumentError("lambda must be nonempty"))
    for value in lambdas
        _validate_nonnegative(value, "lambda")
    end

    nrows = length(b)
    nfeatures = size(A, 2)
    sample_size = max(1, min(nrows, round(Int, subsample_fraction * nrows)))
    selection_counts = zeros(Int, nfeatures, length(lambdas))

    for _ in 1:samples
        rows = Random.shuffle(rng, collect(1:nrows))[1:sample_size]
        sample_A = A[rows, :]
        sample_b = b[rows]
        for (index, value) in pairs(lambdas)
            info = _solve_lasso_sample(
                sample_A,
                sample_b,
                value,
                algorithm,
                nonnegative;
                x0,
                step,
                rho,
                kwargs...,
            )
            selection_counts[:, index] .+= abs.(info.x) .> selection_tol
        end
    end

    probabilities = selection_counts ./ samples
    support = probabilities .>= threshold
    return (
        lambdas=lambdas,
        selection_probability=probabilities,
        stable_support=support,
        selection_counts=selection_counts,
        threshold=threshold,
        sample_size=sample_size,
    )
end

"""
    lasso_lambda_path(A, b, lambdas; kwargs...)

Solve LASSO across a lambda path and report coefficient paths, residual norms,
L1 norms, and active-set indicators. Lambdas are solved in the supplied order
with warm starts after the first solve.
"""
function lasso_lambda_path(A, b, lambdas; kwargs...)
    return _lasso_lambda_path(A, b, lambdas, false; kwargs...)
end

"""
    nonnegative_lasso_lambda_path(A, b, lambdas; kwargs...)

Lambda-path sensitivity for nonnegative penalized LASSO.
"""
function nonnegative_lasso_lambda_path(A, b, lambdas; kwargs...)
    return _lasso_lambda_path(A, b, lambdas, true; kwargs...)
end

function _lasso_lambda_path(
    A,
    b,
    lambdas,
    nonnegative;
    algorithm=:fista,
    selection_tol=1e-8,
    x0=nothing,
    step=nothing,
    rho=1.0,
    kwargs...,
)
    _validate_problem(A, b)
    _validate_nonnegative(selection_tol, "selection_tol")
    _validate_uncertainty_kwargs(kwargs)
    lambda_grid = float.(collect(lambdas))
    !isempty(lambda_grid) || throw(ArgumentError("lambdas must be nonempty"))
    for lambda in lambda_grid
        _validate_nonnegative(lambda, "lambda")
    end

    solver = _noise_sweep_solver(A, b, algorithm; x0, step, rho)
    nfeatures = size(A, 2)
    coefficients = zeros(float(promote_type(eltype(A), eltype(b))), nfeatures, length(lambda_grid))
    residuals = zeros(eltype(coefficients), length(lambda_grid))
    l1_norms = zeros(eltype(coefficients), length(lambda_grid))
    infos = NamedTuple[]

    for (index, lambda) in pairs(lambda_grid)
        info = _noise_sweep_solve(
            solver,
            lambda,
            algorithm,
            nonnegative;
            warm_start=index > 1,
            kwargs...,
        )
        info = merge(info, (x=copy(info.x),))
        coefficients[:, index] = info.x
        residuals[index] = norm(A * info.x - b)
        l1_norms[index] = norm(info.x, 1)
        push!(infos, info)
    end

    active = abs.(coefficients) .> selection_tol
    activation_min_lambda = Vector{Union{Nothing,eltype(lambda_grid)}}(nothing, nfeatures)
    activation_max_lambda = Vector{Union{Nothing,eltype(lambda_grid)}}(nothing, nfeatures)
    for feature in 1:nfeatures
        active_lambdas = lambda_grid[active[feature, :]]
        if !isempty(active_lambdas)
            activation_min_lambda[feature] = minimum(active_lambdas)
            activation_max_lambda[feature] = maximum(active_lambdas)
        end
    end

    return (
        lambdas=lambda_grid,
        coefficients=coefficients,
        residuals=residuals,
        l1_norms=l1_norms,
        active=active,
        activation_min_lambda=activation_min_lambda,
        activation_max_lambda=activation_max_lambda,
        infos=infos,
    )
end

"""
    debiased_lasso_refit(A, b, x; support_tol=1e-8, noise_variance=nothing)

Refit least squares on the support selected by a LASSO solution and report an
approximate covariance and standard errors conditional on that support. This is
a useful diagnostic, but it does not account for support-selection uncertainty.
"""
function debiased_lasso_refit(
    A,
    b,
    x;
    support_tol=1e-8,
    noise_variance=nothing,
    positive=false,
)
    _validate_problem(A, b)
    length(x) == size(A, 2) || throw(DimensionMismatch("x must have length size(A, 2)"))
    _validate_nonnegative(support_tol, "support_tol")
    support = findall(abs.(x) .> support_tol)
    coefficients = zeros(float(promote_type(eltype(A), eltype(b))), size(A, 2))
    covariance = zeros(eltype(coefficients), size(A, 2), size(A, 2))
    standard_error = zeros(eltype(coefficients), size(A, 2))

    if isempty(support)
        residual = copy(b)
        sigma2 = isnothing(noise_variance) ? sum(abs2, residual) / length(b) : float(noise_variance)
        return (
            x=coefficients,
            support=support,
            residual=residual,
            noise_variance=sigma2,
            covariance=covariance,
            standard_error=standard_error,
        )
    end

    A_support = A[:, support]
    refit = A_support \ b
    if positive
        refit = max.(refit, zero(eltype(refit)))
    end
    coefficients[support] = refit
    residual = b - A * coefficients
    degrees_of_freedom = max(length(b) - length(support), 1)
    sigma2 = isnothing(noise_variance) ? sum(abs2, residual) / degrees_of_freedom : float(noise_variance)
    _validate_nonnegative(sigma2, "noise_variance")
    support_covariance = sigma2 .* pinv(transpose(A_support) * A_support)
    covariance[support, support] = support_covariance
    standard_error[support] = sqrt.(max.(diag(support_covariance), zero(eltype(support_covariance))))

    return (
        x=coefficients,
        support=support,
        residual=residual,
        noise_variance=sigma2,
        covariance=covariance,
        standard_error=standard_error,
    )
end

function _solve_lasso_sample(
    A,
    b,
    lambda,
    algorithm,
    nonnegative;
    x0,
    step,
    rho,
    kwargs...,
)
    solver = _noise_sweep_solver(A, b, algorithm; x0, step, rho)
    info = _noise_sweep_solve(
        solver,
        lambda,
        algorithm,
        nonnegative;
        warm_start=false,
        kwargs...,
    )
    return merge(info, (x=copy(info.x),))
end

function _uncertainty_summary(A, coefficient_draws; confidence_level, selection_tol)
    lower_probability = (1 - confidence_level) / 2
    upper_probability = 1 - lower_probability
    coefficient_mean = vec(sum(coefficient_draws; dims=2)) ./ size(coefficient_draws, 2)
    coefficient_std = _row_standard_deviation(coefficient_draws, coefficient_mean)
    lower = _row_quantile(coefficient_draws, lower_probability)
    upper = _row_quantile(coefficient_draws, upper_probability)
    selection_probability = vec(sum(abs.(coefficient_draws) .> selection_tol; dims=2)) ./ size(coefficient_draws, 2)
    prediction_draws = A * coefficient_draws
    prediction_mean = vec(sum(prediction_draws; dims=2)) ./ size(prediction_draws, 2)
    prediction_std = _row_standard_deviation(prediction_draws, prediction_mean)

    return (
        coefficient_draws=coefficient_draws,
        coefficient_mean=coefficient_mean,
        coefficient_std=coefficient_std,
        coefficient_interval=(lower=lower, upper=upper),
        selection_probability=selection_probability,
        prediction_mean=prediction_mean,
        prediction_std=prediction_std,
        confidence_level=confidence_level,
        selection_tol=selection_tol,
    )
end

function _row_standard_deviation(draws, means)
    nsamples = size(draws, 2)
    deviations = zeros(eltype(draws), size(draws, 1))
    if nsamples <= 1
        return deviations
    end
    for row in axes(draws, 1)
        for sample in axes(draws, 2)
            deviations[row] += (draws[row, sample] - means[row])^2
        end
        deviations[row] = sqrt(deviations[row] / (nsamples - 1))
    end
    return deviations
end

function _row_quantile(draws, probability)
    quantiles = zeros(eltype(draws), size(draws, 1))
    scratch = Vector{eltype(draws)}(undef, size(draws, 2))
    for row in axes(draws, 1)
        copyto!(scratch, view(draws, row, :))
        sort!(scratch)
        quantiles[row] = _sorted_quantile(scratch, probability)
    end
    return quantiles
end

function _sorted_quantile(sorted_values, probability)
    length(sorted_values) == 1 && return first(sorted_values)
    position = 1 + probability * (length(sorted_values) - 1)
    lower = floor(Int, position)
    upper = ceil(Int, position)
    lower == upper && return sorted_values[lower]
    weight = position - lower
    return (1 - weight) * sorted_values[lower] + weight * sorted_values[upper]
end

function _noise_sigma(noise_sigma, noise_norm, n, method)
    if method === :bootstrap
        return nothing
    end
    if isnothing(noise_sigma) && isnothing(noise_norm)
        throw(ArgumentError("noise_sigma or noise_norm is required"))
    elseif !isnothing(noise_sigma) && !isnothing(noise_norm)
        throw(ArgumentError("pass only one of noise_sigma or noise_norm"))
    elseif !isnothing(noise_sigma)
        _validate_nonnegative(noise_sigma, "noise_sigma")
        return float(noise_sigma)
    end
    _validate_nonnegative(noise_norm, "noise_norm")
    return float(noise_norm) / sqrt(n)
end

function _validate_confidence_level(confidence_level)
    0 < confidence_level < 1 ||
        throw(ArgumentError("confidence_level must be between 0 and 1"))
    return nothing
end

function _validate_probability(value, name)
    0 <= value <= 1 || throw(ArgumentError("$(name) must be between 0 and 1"))
    return nothing
end

function _validate_uncertainty_kwargs(kwargs)
    haskey(kwargs, :warm_start) &&
        throw(ArgumentError("warm_start is managed by uncertainty routines"))
    haskey(kwargs, :return_info) &&
        throw(ArgumentError("return_info is managed by uncertainty routines"))
    return nothing
end
