export debiased_lasso,
    lasso_refit,
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

Estimate uncertainty by nonparametric row bootstrap. Each sample draws
`length(b)` rows from `(A, b)` with replacement, solves LASSO at the same
`lambda`, and stores the coefficient vector. This captures sensitivity to the
observed measurement set and is useful when rows can be treated as exchangeable
measurements.

Important keywords:
- `samples`: number of bootstrap solves.
- `algorithm`: `:fista` or `:admm`.
- `rng`: random-number generator for reproducible resampling.
- `confidence_level`: coefficient quantile interval level, default `0.95`.
- `selection_tol`: absolute coefficient threshold for active-set probability.

Returns coefficient draws, coefficient mean/std, coefficient quantile intervals,
selection probabilities, and prediction mean/std on the original rows.
"""
function lasso_bootstrap_uncertainty(A, b, lambda; kwargs...)
    return _lasso_resampling_uncertainty(A, b, lambda, false, :bootstrap; kwargs...)
end

"""
    nonnegative_lasso_bootstrap_uncertainty(A, b, lambda; samples=100, kwargs...)

Bootstrap uncertainty for nonnegative penalized LASSO. This is the nonnegative
counterpart of `lasso_bootstrap_uncertainty`: every bootstrap resample is solved
with the nonnegative LASSO solver, and selection probabilities are computed from
nonnegative coefficient draws.
"""
function nonnegative_lasso_bootstrap_uncertainty(A, b, lambda; kwargs...)
    return _lasso_resampling_uncertainty(A, b, lambda, true, :bootstrap; kwargs...)
end

"""
    lasso_noise_perturbation_uncertainty(A, b, lambda; noise_sigma, samples=100, kwargs...)

Estimate uncertainty from an assumed additive Gaussian noise model. Each sample
solves LASSO with `b_sample = b + noise_sigma * randn(...)`. If only the L2 norm
of the noise is known, pass `noise_norm`; it is converted to
`noise_sigma = noise_norm / sqrt(length(b))`.

Use this when the noise model is more trustworthy than row resampling. The
returned summary has the same fields as the bootstrap routine: coefficient
mean/std, quantile intervals, selection probabilities, and prediction summaries.
"""
function lasso_noise_perturbation_uncertainty(A, b, lambda; kwargs...)
    return _lasso_resampling_uncertainty(A, b, lambda, false, :noise; kwargs...)
end

"""
    nonnegative_lasso_noise_perturbation_uncertainty(A, b, lambda; noise_sigma, samples=100, kwargs...)

Noise-perturbation uncertainty for nonnegative penalized LASSO. Each perturbed
right-hand side is solved with the nonnegative LASSO solver, so coefficient draws
respect the positivity constraint.
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

Estimate active-set stability by repeatedly solving LASSO on random row subsets.
A coefficient is counted as selected when `abs(x[j]) > selection_tol`. The main
output is `selection_probability`, a coefficient-by-lambda matrix.

Pass either a single `lambda` or a vector of lambdas. `stable_support` marks
entries whose selection probability is at least `threshold` (default `0.8`). This
is useful when the support is more important than classical standard errors.
"""
function lasso_stability_selection(A, b, lambda; kwargs...)
    return _lasso_stability_selection(A, b, lambda, false; kwargs...)
end

"""
    nonnegative_lasso_stability_selection(A, b, lambda; samples=100, subsample_fraction=0.5, kwargs...)

Stability selection for nonnegative penalized LASSO. Random subsets are solved
with the nonnegative solver, and selection probabilities summarize how often
each nonnegative coefficient remains active.
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

Measure sensitivity to the regularization strength by solving LASSO over a
user-supplied lambda path. Solves are warm-started in the supplied order, so pass
nearby lambdas in the order you want to inspect, often from large to small.

Returns coefficient paths, residual norms, L1 norms, active-set indicators, and
for each coefficient the minimum/maximum lambda values where it was active.
"""
function lasso_lambda_path(A, b, lambdas; kwargs...)
    return _lasso_lambda_path(A, b, lambdas, false; kwargs...)
end

"""
    nonnegative_lasso_lambda_path(A, b, lambdas; kwargs...)

Lambda-path sensitivity for nonnegative penalized LASSO. This is useful for
checking whether positive components persist across nearby lambda choices or only
appear in a narrow tuning interval.
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
    debiased_lasso(A, b, lambda; x_hat=nothing, M=nothing, noise_variance=nothing, kwargs...)

Construct the desparsified/debiased LASSO estimator

`x_debiased = x_hat + (1 / m) * M * A' * (b - A * x_hat)`

where `m = length(b)` and `x_hat` is either supplied or computed by solving
LASSO at `lambda`. By default, `M = pinv(A' * A / m)`, which is a practical
choice when an explicit matrix pseudo-inverse is affordable. Supplying `M` lets
callers use a custom approximate inverse of the empirical Gram matrix.

The returned covariance follows the Gaussian term in the debiased estimator:

`covariance = noise_variance * M * A' * A * M' / m^2`.

If `noise_variance` is omitted, it is estimated from the LASSO residual using the
active-set size as degrees of freedom. Alternatively pass `noise_sigma` or
`noise_norm`. The returned `bias_operator = I - M * (A' * A / m)` and
`bias_operator_infnorm` summarize the remainder-control term; the true remainder
still depends on the unknown recovery error.

Important keywords:
- `x_hat`: existing LASSO estimate to debias.
- `positive`: compute `x_hat` with nonnegative LASSO when `x_hat` is omitted.
- `algorithm`: `:fista` or `:admm` for computing `x_hat`.
- `M`: custom approximate inverse of `A' * A / m`.
- `noise_variance`, `noise_sigma`, or `noise_norm`: measurement-noise scale.

Returns `x`, `x_hat`, `covariance`, `standard_error`, `residual`, noise variance,
`M`, empirical Gram matrix, and bias-operator diagnostics.
"""
function debiased_lasso(
    A,
    b,
    lambda;
    x_hat=nothing,
    M=nothing,
    noise_variance=nothing,
    noise_sigma=nothing,
    noise_norm=nothing,
    positive=false,
    algorithm=:fista,
    support_tol=1e-8,
    x0=nothing,
    step=nothing,
    rho=1.0,
    kwargs...,
)
    _validate_problem(A, b)
    _validate_nonnegative(lambda, "lambda")
    _validate_nonnegative(support_tol, "support_tol")
    _validate_uncertainty_kwargs(kwargs)

    m = length(b)
    x_lasso = isnothing(x_hat) ?
              _solve_lasso_sample(
                  A,
                  b,
                  lambda,
                  algorithm,
                  positive;
                  x0,
                  step,
                  rho,
                  kwargs...,
              ).x :
              copy(float.(x_hat))
    length(x_lasso) == size(A, 2) ||
        throw(DimensionMismatch("x_hat must have length size(A, 2)"))

    gram = adjoint(A) * A
    empirical_gram = gram ./ m
    correction_matrix = isnothing(M) ? pinv(empirical_gram) : float.(M)
    size(correction_matrix) == (size(A, 2), size(A, 2)) ||
        throw(DimensionMismatch("M must have size (size(A, 2), size(A, 2))"))

    residual = b - A * x_lasso
    x = x_lasso + (correction_matrix * (adjoint(A) * residual)) ./ m
    sigma2 = _debiased_noise_variance(
        residual,
        x_lasso,
        m;
        support_tol,
        noise_variance,
        noise_sigma,
        noise_norm,
    )
    covariance = sigma2 .* correction_matrix * gram * adjoint(correction_matrix) ./ (m^2)
    standard_error = sqrt.(max.(diag(covariance), zero(eltype(covariance))))
    bias_operator = Matrix{eltype(covariance)}(I, size(A, 2), size(A, 2)) -
                    correction_matrix * empirical_gram

    return (
        x=x,
        x_hat=x_lasso,
        residual=residual,
        noise_variance=sigma2,
        covariance=covariance,
        standard_error=standard_error,
        M=correction_matrix,
        empirical_gram=empirical_gram,
        bias_operator=bias_operator,
        bias_operator_infnorm=norm(bias_operator, Inf),
    )
end

function _debiased_noise_variance(
    residual,
    x_hat,
    m;
    support_tol,
    noise_variance,
    noise_sigma,
    noise_norm,
)
    supplied = count(!isnothing, (noise_variance, noise_sigma, noise_norm))
    supplied <= 1 || throw(ArgumentError("pass only one of noise_variance, noise_sigma, or noise_norm"))
    if !isnothing(noise_variance)
        _validate_nonnegative(noise_variance, "noise_variance")
        return float(noise_variance)
    elseif !isnothing(noise_sigma)
        _validate_nonnegative(noise_sigma, "noise_sigma")
        return float(noise_sigma)^2
    elseif !isnothing(noise_norm)
        _validate_nonnegative(noise_norm, "noise_norm")
        return float(noise_norm)^2 / m
    end

    active = count(abs.(x_hat) .> support_tol)
    degrees_of_freedom = max(m - active, 1)
    return sum(abs2, residual) / degrees_of_freedom
end

"""
    lasso_refit(A, b, x; support_tol=1e-8, noise_variance=nothing, positive=false)

Refit on the support selected by a LASSO solution. For signed models this uses
ordinary least squares on `support = findall(abs.(x) .> support_tol)`. With
`positive=true`, the refit uses an active-set nonnegative least-squares solve on
that selected support; coefficients that cannot remain positive are dropped and
reported through `support`, while the original LASSO-selected support is returned
as `original_support`.

The covariance and standard errors are computed only on the final refit support,
using `noise_variance` if supplied or the residual variance estimate otherwise.
These are conditional diagnostics: they do not include uncertainty from the
support-selection step itself.
"""
function lasso_refit(
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
            original_support=support,
            residual=residual,
            noise_variance=sigma2,
            covariance=covariance,
            standard_error=standard_error,
        )
    end

    A_support = A[:, support]
    if positive
        refit, active_support_indices = _nnls_active_set(A_support, b)
        final_support = support[active_support_indices]
    else
        refit = A_support \ b
        final_support = support
    end

    coefficients[support] = refit
    residual = b - A * coefficients
    degrees_of_freedom = max(length(b) - length(final_support), 1)
    sigma2 = isnothing(noise_variance) ? sum(abs2, residual) / degrees_of_freedom : float(noise_variance)
    _validate_nonnegative(sigma2, "noise_variance")

    if !isempty(final_support)
        A_active = A[:, final_support]
        active_covariance = sigma2 .* pinv(transpose(A_active) * A_active)
        covariance[final_support, final_support] = active_covariance
        standard_error[final_support] = sqrt.(max.(diag(active_covariance), zero(eltype(active_covariance))))
    end

    return (
        x=coefficients,
        support=final_support,
        original_support=support,
        residual=residual,
        noise_variance=sigma2,
        covariance=covariance,
        standard_error=standard_error,
    )
end

function _nnls_active_set(A, b; tol=100 * eps(float(promote_type(eltype(A), eltype(b)))), maxiter=30 * size(A, 2))
    n = size(A, 2)
    T = float(promote_type(eltype(A), eltype(b)))
    x = zeros(T, n)
    passive = falses(n)
    gradient = transpose(A) * (b - A * x)
    iterations = 0

    while any((.!passive) .& (gradient .> tol))
        iterations += 1
        iterations <= maxiter || break
        candidate = argmax(ifelse.(passive, typemin(T), gradient))
        passive[candidate] = true

        while true
            passive_indices = findall(passive)
            trial = zeros(T, n)
            trial[passive_indices] = A[:, passive_indices] \ b
            if all(trial[passive_indices] .> tol)
                x = trial
                break
            end

            blocking = passive_indices[trial[passive_indices] .<= tol]
            ratios = [x[j] / (x[j] - trial[j]) for j in blocking if x[j] > tol]
            if isempty(ratios)
                for j in blocking
                    passive[j] = false
                    x[j] = zero(T)
                end
                break
            end

            alpha = minimum(ratios)
            x .= x .+ alpha .* (trial .- x)
            for j in passive_indices
                if x[j] <= tol
                    passive[j] = false
                    x[j] = zero(T)
                end
            end
        end

        gradient = transpose(A) * (b - A * x)
    end

    x[abs.(x) .<= tol] .= zero(T)
    return x, findall(x .> tol)
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
