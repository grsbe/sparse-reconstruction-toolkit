export bisection_parameter_sweep,
    distance_score,
    parameter_sweep,
    stop_when_above,
    stop_when_below,
    stop_when_within

"""
    parameter_sweep(candidates, solve; stop, score, maxiter, snapshot)

Run a solver-agnostic sweep over `candidates`. `solve(candidate)` performs one
trial and may call any solver. `score(result)` is minimized to select the best
trial, while `stop(result)` can end the sweep early.

Use `snapshot` when the solver returns mutable state that will be reused by warm
starts, for example `snapshot = r -> merge(r, (x=copy(r.x),))`.
"""
function parameter_sweep(
    candidates,
    solve;
    stop=_ -> false,
    score=_ -> 0.0,
    maxiter=nothing,
    snapshot=identity,
)
    _validate_optional_maxiter(maxiter)

    trials = NamedTuple[]
    best = nothing
    stopped = false

    for (iteration, candidate) in enumerate(candidates)
        if !isnothing(maxiter) && iteration > maxiter
            break
        end

        result = snapshot(solve(candidate))
        value = score(result)
        trial = (
            iteration=iteration,
            candidate=candidate,
            result=result,
            score=value,
            stopped=stop(result),
        )
        push!(trials, trial)

        if isnothing(best) || value < best.score
            best = trial
        end

        if trial.stopped
            stopped = true
            break
        end
    end

    !isempty(trials) || throw(ArgumentError("candidates must be nonempty"))

    return (
        candidate=best.candidate,
        result=best.result,
        score=best.score,
        stopped=stopped,
        iterations=length(trials),
        trials=trials,
    )
end

"""
    bisection_parameter_sweep(solve, low, high; target, metric, steps=20, kwargs...)

Run a solver-agnostic bisection sweep for a scalar parameter. `solve(value)`
runs one trial and `metric(result)` returns the model-data mismatch, likelihood,
or other scalar quantity being matched to `target`.

By default the metric is assumed to increase with the swept parameter. Set
`increasing=false` when larger parameters make the metric smaller.
"""
function bisection_parameter_sweep(
    solve,
    low,
    high;
    target,
    metric,
    steps=20,
    increasing=true,
    abstol=0.0,
    reltol=0.0,
    stop=nothing,
    score=nothing,
    snapshot=identity,
)
    _validate_bisection_bounds(low, high)
    steps isa Integer && steps > 0 ||
        throw(ArgumentError("steps must be a positive integer"))

    local_stop = isnothing(stop) ?
                 stop_when_within(metric, target; abstol, reltol) :
                 stop
    local_score = isnothing(score) ? distance_score(metric, target) : score

    trials = NamedTuple[]
    best = nothing
    stopped = false
    current_low = float(low)
    current_high = float(high)

    for iteration in 1:steps
        candidate = (current_low + current_high) / 2
        result = snapshot(solve(candidate))
        value = metric(result)
        candidate_score = local_score(result)
        trial = (
            iteration=iteration,
            candidate=candidate,
            result=result,
            metric=value,
            score=candidate_score,
            low=current_low,
            high=current_high,
            stopped=local_stop(result),
        )
        push!(trials, trial)

        if isnothing(best) || candidate_score < best.score
            best = trial
        end

        if trial.stopped
            stopped = true
            break
        end

        if increasing
            if value > target
                current_high = candidate
            else
                current_low = candidate
            end
        else
            if value > target
                current_low = candidate
            else
                current_high = candidate
            end
        end
    end

    return (
        candidate=best.candidate,
        result=best.result,
        metric=best.metric,
        score=best.score,
        stopped=stopped,
        iterations=length(trials),
        low=current_low,
        high=current_high,
        trials=trials,
    )
end

"""
    distance_score(metric, target)

Return a scoring function that prefers results whose `metric(result)` is closest
to `target`.
"""
distance_score(metric, target) = result -> abs(metric(result) - target)

"""
    stop_when_within(metric, target; abstol=0, reltol=0)

Return a stop rule that accepts a result once `metric(result)` is within the
absolute/relative tolerance of `target`.
"""
function stop_when_within(metric, target; abstol=0.0, reltol=0.0)
    tolerance = _sweep_tolerance(target, abstol, reltol)
    return result -> abs(metric(result) - target) <= tolerance
end

"""
    stop_when_below(metric, target; abstol=0, reltol=0)

Return a stop rule that accepts a result once `metric(result)` is at or below
`target` within tolerance.
"""
function stop_when_below(metric, target; abstol=0.0, reltol=0.0)
    tolerance = _sweep_tolerance(target, abstol, reltol)
    return result -> metric(result) <= target + tolerance
end

"""
    stop_when_above(metric, target; abstol=0, reltol=0)

Return a stop rule that accepts a result once `metric(result)` is at or above
`target` within tolerance.
"""
function stop_when_above(metric, target; abstol=0.0, reltol=0.0)
    tolerance = _sweep_tolerance(target, abstol, reltol)
    return result -> metric(result) >= target - tolerance
end

function _validate_optional_maxiter(maxiter)
    if !isnothing(maxiter)
        maxiter isa Integer && maxiter > 0 ||
            throw(ArgumentError("maxiter must be a positive integer"))
    end
    return nothing
end

function _validate_bisection_bounds(low, high)
    isfinite(low) || throw(ArgumentError("low must be finite"))
    isfinite(high) || throw(ArgumentError("high must be finite"))
    low < high || throw(ArgumentError("low must be less than high"))
    return nothing
end

function _sweep_tolerance(target, abstol, reltol)
    _validate_nonnegative(abstol, "abstol")
    _validate_nonnegative(reltol, "reltol")
    scale = max(abs(float(target)), eps(float(target)))
    return float(abstol) + float(reltol) * scale
end
