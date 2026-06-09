export bisection_parameter_sweep,
    bracketed_parameter_sweep,
    distance_score,
    multiplicative_parameter_sweep,
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
            phase=:bisection,
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
    multiplicative_parameter_sweep(solve, start; target, metric, factor=2, kwargs...)

Tune one positive scalar parameter with multiplicative steps. The sweep starts at
`start`, moves in the direction that should bring `metric(result)` toward
`target`, and reverses direction whenever the target is crossed. Each crossing
replaces `factor` with `sqrt(factor)`, giving finer multiplicative refinement
without switching to bisection.

By default the metric is assumed to increase with the parameter. Set
`increasing=false` when larger parameters make the metric smaller.
"""
function multiplicative_parameter_sweep(
    solve,
    start;
    target,
    metric,
    factor=2.0,
    steps=50,
    increasing=true,
    abstol=0.0,
    reltol=1e-3,
    stop=nothing,
    score=nothing,
    snapshot=identity,
)
    _validate_positive(start, "start")
    _validate_positive(factor, "factor")
    factor > 1 || throw(ArgumentError("factor must be greater than 1"))
    steps isa Integer && steps > 0 ||
        throw(ArgumentError("steps must be a positive integer"))

    local_stop = isnothing(stop) ?
                 stop_when_within(metric, target; abstol, reltol) :
                 stop
    local_score = isnothing(score) ? distance_score(metric, target) : score

    trials = NamedTuple[]
    candidate = float(start)
    result = snapshot(solve(candidate))
    value = metric(result)
    candidate_score = local_score(result)
    stopped = local_stop(result)
    current_factor = float(factor)
    search_up = increasing ? value < target : value > target
    trial = (
        iteration=1,
        candidate=candidate,
        result=result,
        metric=value,
        score=candidate_score,
        factor=current_factor,
        direction=search_up ? :up : :down,
        crossed=false,
        stopped=stopped,
    )
    push!(trials, trial)
    best = trial

    if stopped
        return _multiplicative_sweep_return(best, true, current_factor, trials)
    end

    previous_value = value
    for iteration in 2:steps
        candidate = search_up ? candidate * current_factor : candidate / current_factor
        result = snapshot(solve(candidate))
        value = metric(result)
        candidate_score = local_score(result)
        stopped = local_stop(result)
        crossed = _target_crossed(previous_value, value, target)

        if crossed
            search_up = !search_up
            current_factor = sqrt(current_factor)
        end

        trial = (
            iteration=iteration,
            candidate=candidate,
            result=result,
            metric=value,
            score=candidate_score,
            factor=current_factor,
            direction=search_up ? :up : :down,
            crossed=crossed,
            stopped=stopped,
        )
        push!(trials, trial)

        if candidate_score < best.score
            best = trial
        end

        if stopped
            return _multiplicative_sweep_return(best, true, current_factor, trials)
        end

        previous_value = value
    end

    return _multiplicative_sweep_return(best, false, current_factor, trials)
end

"""
    bracketed_parameter_sweep(solve, start; target, metric, factor=2, kwargs...)

Find a scalar parameter by first expanding exponentially from `start` until
`metric(result)` crosses `target`, then refining the discovered bracket with
`bisection_parameter_sweep`.

This is usually more ergonomic than manually choosing `low` and `high`. By
default the metric is assumed to increase with the parameter; set
`increasing=false` when larger parameters make the metric smaller.

Set `on_bracket_failure=:return` to get the best expansion trial back instead of
throwing an error when no crossing is found.
"""
function bracketed_parameter_sweep(
    solve,
    start;
    target,
    metric,
    factor=2.0,
    bracket_steps=20,
    bisection_steps=20,
    increasing=true,
    abstol=0.0,
    reltol=0.0,
    stop=nothing,
    score=nothing,
    snapshot=identity,
    on_bracket_failure=:error,
)
    _validate_positive(start, "start")
    _validate_positive(factor, "factor")
    factor > 1 || throw(ArgumentError("factor must be greater than 1"))
    bracket_steps isa Integer && bracket_steps > 0 ||
        throw(ArgumentError("bracket_steps must be a positive integer"))
    bisection_steps isa Integer && bisection_steps > 0 ||
        throw(ArgumentError("bisection_steps must be a positive integer"))
    on_bracket_failure in (:error, :return) ||
        throw(ArgumentError("on_bracket_failure must be :error or :return"))

    local_stop = isnothing(stop) ?
                 stop_when_within(metric, target; abstol, reltol) :
                 stop
    local_score = isnothing(score) ? distance_score(metric, target) : score

    bracket_trials = NamedTuple[]
    candidate = float(start)
    result = snapshot(solve(candidate))
    value = metric(result)
    candidate_score = local_score(result)
    stopped = local_stop(result)
    first_trial = (
        phase=:bracket,
        iteration=1,
        candidate=candidate,
        result=result,
        metric=value,
        score=candidate_score,
        low=candidate,
        high=candidate,
        stopped=stopped,
    )
    push!(bracket_trials, first_trial)
    best = first_trial

    if stopped
        return _bracketed_sweep_return(best, true, true, candidate, candidate, bracket_trials, nothing)
    end

    previous_candidate = candidate
    previous_value = value
    search_up = increasing ? value < target : value > target

    for iteration in 2:(bracket_steps + 1)
        candidate = search_up ? previous_candidate * factor : previous_candidate / factor
        result = snapshot(solve(candidate))
        value = metric(result)
        candidate_score = local_score(result)
        stopped = local_stop(result)
        low = min(previous_candidate, candidate)
        high = max(previous_candidate, candidate)
        trial = (
            phase=:bracket,
            iteration=iteration,
            candidate=candidate,
            result=result,
            metric=value,
            score=candidate_score,
            low=low,
            high=high,
            stopped=stopped,
        )
        push!(bracket_trials, trial)

        if candidate_score < best.score
            best = trial
        end

        bracket_found = _target_crossed(previous_value, value, target)
        if stopped
            return _bracketed_sweep_return(best, true, true, low, high, bracket_trials, nothing)
        elseif bracket_found
            bisection = bisection_parameter_sweep(
                solve,
                low,
                high;
                target,
                metric,
                steps=bisection_steps,
                increasing,
                abstol,
                reltol,
                stop,
                score,
                snapshot,
            )
            if bisection.score < best.score
                best = (
                    phase=:bisection,
                    iteration=length(bracket_trials) + bisection.iterations,
                    candidate=bisection.candidate,
                    result=bisection.result,
                    metric=bisection.metric,
                    score=bisection.score,
                    low=bisection.low,
                    high=bisection.high,
                    stopped=bisection.stopped,
                )
            end
            return _bracketed_sweep_return(
                best,
                true,
                bisection.stopped,
                low,
                high,
                bracket_trials,
                bisection,
            )
        end

        previous_candidate = candidate
        previous_value = value
    end

    if on_bracket_failure === :error
        throw(ArgumentError("failed to bracket target after $(bracket_steps) expansion steps"))
    end

    return _bracketed_sweep_return(
        best,
        false,
        false,
        minimum(trial.candidate for trial in bracket_trials),
        maximum(trial.candidate for trial in bracket_trials),
        bracket_trials,
        nothing,
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


function _multiplicative_sweep_return(best, stopped, factor, trials)
    return (
        candidate=best.candidate,
        result=best.result,
        metric=best.metric,
        score=best.score,
        stopped=stopped,
        iterations=length(trials),
        factor=factor,
        trials=trials,
    )
end

function _target_crossed(left_value, right_value, target)
    left_gap = left_value - target
    right_gap = right_value - target
    return left_gap == 0 || right_gap == 0 || sign(left_gap) != sign(right_gap)
end

function _bracketed_sweep_return(best, bracket_found, stopped, low, high, bracket_trials, bisection)
    trials = isnothing(bisection) ?
             bracket_trials :
             vcat(bracket_trials, bisection.trials)
    return (
        candidate=best.candidate,
        result=best.result,
        metric=best.metric,
        score=best.score,
        stopped=stopped,
        bracket_found=bracket_found,
        iterations=length(trials),
        low=low,
        high=high,
        bracket_trials=bracket_trials,
        bisection=bisection,
        trials=trials,
    )
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
