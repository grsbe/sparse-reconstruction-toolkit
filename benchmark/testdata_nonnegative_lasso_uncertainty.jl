using JLD2
using LinearAlgebra
using Printf
using Random
using Statistics
using SparseReconstructionToolkit

const HAS_CAIROMAKIE = try
    @eval import CairoMakie
    true
catch err
    @warn "CairoMakie is not available; skipping PNG plots" exception = (err, catch_backtrace())
    false
end

const DATA_DIR = joinpath(@__DIR__, "..", "testdata")
const PLOT_DIR = joinpath(@__DIR__, "..", "benchmarkplots")
const BLAS_THREADS = parse(Int, get(ENV, "SRT_BLAS_THREADS", "32"))
const SNRS = parse.(Int, split(get(ENV, "SRT_UNCERTAINTY_SNRS", "30,15"), ","))
const SEARCH_STEPS = parse(Int, get(ENV, "SRT_SEARCH_STEPS", "10"))
const UNCERTAINTY_SAMPLES = parse(Int, get(ENV, "SRT_UNCERTAINTY_SAMPLES", "20"))
const LAMBDA_PATH_STEPS = parse(Int, get(ENV, "SRT_LAMBDA_PATH_STEPS", "9"))
const FISTA_MAXITER = parse(Int, get(ENV, "SRT_FISTA_MAXITER", "2000"))
const SOLVER_ABSTOL = parse(Float64, get(ENV, "SRT_SOLVER_ABSTOL", "1e-7"))
const SOLVER_RELTOL = parse(Float64, get(ENV, "SRT_SOLVER_RELTOL", "1e-5"))
const SELECTION_TOL = parse(Float64, get(ENV, "SRT_SELECTION_TOL", "1e-8"))

load_single(name) = JLD2.load(joinpath(DATA_DIR, name))["single_stored_object"]
relative_error(x, x_true) = norm(x - x_true) / max(norm(x_true), eps())
noise_sigma(noise, y) = noise / sqrt(length(y))
positive_lambda_max(A, y) = max(maximum(transpose(A) * y), eps())
support(x) = abs.(x) .> SELECTION_TOL

function timed(f)
    t0 = time_ns()
    result = f()
    return result, (time_ns() - t0) / 1e9
end

function residual(A, y, x)
    return norm(A * x - y)
end

function lasso_objective(A, y, lambda, x)
    return 0.5 * norm(A * x - y)^2 + lambda * norm(x, 1)
end

function solve_noise_matched_lasso(A, y, noise, step)
    solver = LassoFISTASolver(A, y; step)
    lambda_low = 0.0
    lambda_high = positive_lambda_max(A, y)
    best = nothing
    total_seconds = 0.0

    for trial in 1:SEARCH_STEPS
        lambda = trial == 1 ? lambda_high / 2 : (lambda_low + lambda_high) / 2
        info, seconds = timed(
            () -> nonnegative_lasso_fista(
                solver,
                lambda;
                warm_start=trial > 1,
                abstol=SOLVER_ABSTOL,
                reltol=SOLVER_RELTOL,
                maxiter=FISTA_MAXITER,
                return_info=true,
            ),
        )
        total_seconds += seconds
        current_residual = residual(A, y, info.x)
        candidate = (
            lambda=lambda,
            info=merge(info, (x=copy(info.x),)),
            residual=current_residual,
            gap=current_residual - noise,
            seconds=seconds,
        )
        if isnothing(best) || abs(candidate.gap) < abs(best.gap)
            best = candidate
        end

        @printf(
            "  search=%2d lambda=%9.3g residual=%9.4g gap=%9.3g iter=%5d converged=%5s time=%7.3f s\n",
            trial,
            lambda,
            current_residual,
            current_residual - noise,
            info.iterations,
            string(info.converged),
            seconds,
        )

        if current_residual > noise
            lambda_high = lambda
        else
            lambda_low = lambda
        end
    end

    return best, total_seconds
end

function as_image(x, image_dims)
    transposed_dims = reverse(image_dims)
    prod(transposed_dims) == length(x) || return nothing
    return permutedims(reshape(x, transposed_dims))
end

function summarize_interval(uncertainty, x_true)
    lower = uncertainty.coefficient_interval.lower
    upper = uncertainty.coefficient_interval.upper
    active = support(x_true)
    inactive = .!active
    coverage = mean((lower .<= x_true) .& (x_true .<= upper))
    active_coverage = any(active) ? mean((lower[active] .<= x_true[active]) .& (x_true[active] .<= upper[active])) : NaN
    inactive_coverage = any(inactive) ? mean((lower[inactive] .<= x_true[inactive]) .& (x_true[inactive] .<= upper[inactive])) : NaN
    return coverage, active_coverage, inactive_coverage
end

function summarize_selection(probability, x_true)
    active = support(x_true)
    inactive = .!active
    return (
        active_mean=any(active) ? mean(probability[active]) : NaN,
        inactive_mean=any(inactive) ? mean(probability[inactive]) : NaN,
        selected_at_50=count(probability .>= 0.5),
        selected_at_80=count(probability .>= 0.8),
    )
end

function write_summary_table(rows)
    mkpath(PLOT_DIR)
    path = joinpath(PLOT_DIR, "testdata_nonnegative_lasso_uncertainty_summary.tsv")
    open(path, "w") do io
        println(
            io,
            join(
                [
                    "snr",
                    "estimator",
                    "seconds",
                    "lambda",
                    "residual",
                    "residual_minus_noise",
                    "relative_x_error",
                    "active_selection_mean",
                    "inactive_selection_mean",
                    "interval_coverage",
                    "active_interval_coverage",
                    "inactive_interval_coverage",
                ],
                '\t',
            ),
        )
        for row in rows
            println(
                io,
                join(
                    [
                        row.snr,
                        row.estimator,
                        row.seconds,
                        row.lambda,
                        row.residual,
                        row.residual_minus_noise,
                        row.relative_x_error,
                        row.active_selection_mean,
                        row.inactive_selection_mean,
                        row.interval_coverage,
                        row.active_interval_coverage,
                        row.inactive_interval_coverage,
                    ],
                    '\t',
                ),
            )
        end
    end
    return path
end

function write_raw_results(results, rows, x_true, nonzero_indices, image_dims)
    mkpath(PLOT_DIR)
    path = joinpath(PLOT_DIR, "testdata_nonnegative_lasso_uncertainty_raw.jld2")
    payload = Dict{String,Any}()
    payload["summary_rows"] = rows
    payload["snrs"] = [result.snr for result in results]
    payload["target"] = x_true
    payload["target_nonzero_indices"] = nonzero_indices
    payload["target_nonzero_values"] = x_true[nonzero_indices]
    payload["image_dims"] = image_dims
    for result in results
        payload["SNR$(result.snr)"] = (
            y=result.y,
            noise_norm=result.noise_norm,
            lambda=result.lambda,
            x_hat=result.x_hat,
            bootstrap=result.bootstrap,
            noise=result.noise,
            stability=result.stability,
            lambda_path=result.path,
            refit=result.refit,
        )
    end
    JLD2.save(path, payload)
    return path
end

function nonzero_peak_indices(x_true)
    return findall(abs.(x_true) .> SELECTION_TOL)
end

function peak_labels(indices, image_dims)
    labels = String[]
    for index in indices
        row_t, col_t = Tuple(CartesianIndices(reverse(image_dims))[index])
        push!(labels, "(x=$(row_t), y=$(col_t))")
    end
    return labels
end

function plot_peak_recovery!(paths, results, x_true, nonzero_indices, image_dims)
    isempty(nonzero_indices) && return nothing
    labels = peak_labels(nonzero_indices, image_dims)
    for result in results
        series = [
            ("truth", x_true[nonzero_indices], zeros(length(nonzero_indices))),
            ("FISTA", result.x_hat[nonzero_indices], zeros(length(nonzero_indices))),
            ("bootstrap", result.bootstrap.coefficient_mean[nonzero_indices], result.bootstrap.coefficient_std[nonzero_indices]),
            ("noise", result.noise.coefficient_mean[nonzero_indices], result.noise.coefficient_std[nonzero_indices]),
            ("refit", result.refit.x[nonzero_indices], result.refit.standard_error[nonzero_indices]),
        ]
        fig = CairoMakie.Figure(size=(1400, 520))
        ax = CairoMakie.Axis(fig[1, 1], xlabel="true nonzero target pixel", ylabel="coefficient", xticklabelrotation=pi / 3)
        width = 0.82 / length(series)
        for (series_index, (name, values, errors)) in pairs(series)
            positions = collect(1:length(nonzero_indices)) .+ (series_index - (length(series) + 1) / 2) * width
            CairoMakie.barplot!(ax, positions, values; width, label=name)
            if any(errors .> 0)
                CairoMakie.errorbars!(ax, positions, values, errors; whiskerwidth=width / 2)
            end
        end
        ax.xticks = (1:length(nonzero_indices), labels)
        CairoMakie.axislegend(ax, position=:rb, nbanks=2)
        path = joinpath(PLOT_DIR, "testdata_peak_recovery_SNR$(result.snr).png")
        CairoMakie.save(path, fig)
        push!(paths, path)
    end
    return nothing
end

function neighborhood_groups(index, image_dims; radius=1)
    row_t, col_t = Tuple(CartesianIndices(reverse(image_dims))[index])
    row = col_t
    col = row_t
    center = Int[]
    axial = Int[]
    diagonal = Int[]
    for dr in -radius:radius, dc in -radius:radius
        rr = row + dr
        cc = col + dc
        1 <= rr <= image_dims[1] || continue
        1 <= cc <= image_dims[2] || continue
        vector_index = LinearIndices(reverse(image_dims))[cc, rr]
        if dr == 0 && dc == 0
            push!(center, vector_index)
        elseif max(abs(dr), abs(dc)) == 1 && abs(dr) + abs(dc) == 1
            push!(axial, vector_index)
        elseif max(abs(dr), abs(dc)) == 1
            push!(diagonal, vector_index)
        end
    end
    return (center=center, axial=axial, diagonal=diagonal)
end

function local_value_components(uncertainty, indices, image_dims)
    center = zeros(length(indices))
    axial = zeros(length(indices))
    diagonal = zeros(length(indices))
    total_std = zeros(length(indices))
    for (i, index) in pairs(indices)
        groups = neighborhood_groups(index, image_dims; radius=1)
        center[i] = sum(uncertainty.coefficient_mean[groups.center])
        axial[i] = sum(uncertainty.coefficient_mean[groups.axial])
        diagonal[i] = sum(uncertainty.coefficient_mean[groups.diagonal])
        total_std[i] = uncertainty.coefficient_std[index]
    end
    return center, axial, diagonal, total_std
end
function plot_peak_spatial_uncertainty!(paths, results, x_true, nonzero_indices, image_dims)
    isempty(nonzero_indices) && return nothing
    labels = peak_labels(nonzero_indices, image_dims)
    peak_values = x_true[nonzero_indices]
    for result in results
        fig = CairoMakie.Figure(size=(1400, 560))
        ax = CairoMakie.Axis(
            fig[1, 1],
            xlabel="true nonzero target pixel",
            ylabel="true coefficient / recovered coefficient sum in 3x3 neighborhood",
            xticklabelrotation=pi / 3,
        )
        methods = [
            ("bootstrap", result.bootstrap),
            ("noise", result.noise),
        ]
        width = 0.26
        colors = (:steelblue, :orange, :seagreen)
        base_positions = collect(1:length(nonzero_indices))
        truth_positions = base_positions .- width
        CairoMakie.barplot!(ax, truth_positions, peak_values; width, color=:gray55, label="true emission")
        for (x, y) in zip(truth_positions, peak_values)
            CairoMakie.text!(ax, x, y; text="truth", rotation=pi / 2, align=(:left, :center), fontsize=10)
        end
        for (method_index, (method, uncertainty)) in pairs(methods)
            positions = base_positions .+ (method_index - 1) * width
            center, axial, diagonal, total_std = local_value_components(uncertainty, nonzero_indices, image_dims)
            total = center .+ axial .+ diagonal
            CairoMakie.barplot!(ax, positions, center; width, color=colors[1], label=method_index == 1 ? "center pixel" : nothing)
            CairoMakie.barplot!(ax, positions, axial; width, offset=center, color=colors[2], label=method_index == 1 ? "4-neighbors" : nothing)
            CairoMakie.barplot!(ax, positions, diagonal; width, offset=center .+ axial, color=colors[3], label=method_index == 1 ? "diagonals" : nothing)
            CairoMakie.errorbars!(ax, positions, total, total_std; whiskerwidth=width / 2)
            for (x, y) in zip(positions, total)
                CairoMakie.text!(ax, x, y; text=method, rotation=pi / 2, align=(:left, :center), fontsize=10)
            end
        end
        ax.xticks = (1:length(nonzero_indices), labels)
        CairoMakie.axislegend(ax, position=:rt)
        path = joinpath(PLOT_DIR, "testdata_peak_spatial_uncertainty_SNR$(result.snr).png")
        CairoMakie.save(path, fig)
        push!(paths, path)
    end
    return nothing
end
function maybe_plot_results(results, rows, x_true, image_dims, nonzero_indices)
    HAS_CAIROMAKIE || return String[]

    mkpath(PLOT_DIR)
    paths = String[]
    snrs = sort(unique([row.snr for row in rows]); rev=true)
    estimators = unique([row.estimator for row in rows])

    fig = CairoMakie.Figure(size=(1100, 420))
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel="estimator", ylabel="relative x error", xticklabelrotation=pi / 6)
    ax2 = CairoMakie.Axis(fig[1, 2], xlabel="estimator", ylabel="seconds", xticklabelrotation=pi / 6, yscale=log10)
    width = 0.8 / max(length(snrs), 1)
    for (snr_index, snr) in pairs(snrs)
        snr_rows = [row for row in rows if row.snr == snr]
        positions = [findfirst(==(row.estimator), estimators) + (snr_index - (length(snrs) + 1) / 2) * width for row in snr_rows]
        CairoMakie.barplot!(ax1, positions, [row.relative_x_error for row in snr_rows]; width, label="SNR $snr")
        CairoMakie.barplot!(ax2, positions, [max(row.seconds, eps()) for row in snr_rows]; width, label="SNR $snr")
    end
    ax1.xticks = (1:length(estimators), estimators)
    ax2.xticks = (1:length(estimators), estimators)
    CairoMakie.axislegend(ax1, position=:lt)
    path = joinpath(PLOT_DIR, "testdata_uncertainty_error_time.png")
    CairoMakie.save(path, fig)
    push!(paths, path)

    for result in results
        image_true = as_image(x_true, image_dims)
        image_lasso = as_image(result.x_hat, image_dims)
        image_bootstrap_std = as_image(result.bootstrap.coefficient_std, image_dims)
        image_noise_std = as_image(result.noise.coefficient_std, image_dims)
        if all(!isnothing, (image_true, image_lasso, image_bootstrap_std, image_noise_std))
            fig = CairoMakie.Figure(size=(1000, 760))
            items = [
                ("target", image_true),
                ("FISTA estimate", image_lasso),
                ("bootstrap std", image_bootstrap_std),
                ("noise perturbation std", image_noise_std),
            ]
            for (index, (title, image)) in pairs(items)
                ax = CairoMakie.Axis(fig[cld(index, 2), mod1(index, 2)], title=title, aspect=CairoMakie.DataAspect())
                CairoMakie.hidedecorations!(ax)
                CairoMakie.heatmap!(ax, rotr90(image); colormap=:viridis)
            end
            path = joinpath(PLOT_DIR, "testdata_uncertainty_maps_SNR$(result.snr).png")
            CairoMakie.save(path, fig)
            push!(paths, path)
        end

        fig = CairoMakie.Figure(size=(950, 420))
        ax = CairoMakie.Axis(fig[1, 1], xlabel="lambda", ylabel="residual norm", xscale=log10)
        CairoMakie.scatterlines!(ax, result.path.lambdas, result.path.residuals)
        CairoMakie.hlines!(ax, [result.noise_norm]; color=:red, linestyle=:dash, label="noise norm")
        CairoMakie.axislegend(ax)
        path = joinpath(PLOT_DIR, "testdata_lambda_path_residuals_SNR$(result.snr).png")
        CairoMakie.save(path, fig)
        push!(paths, path)
    end

    plot_peak_recovery!(paths, results, x_true, nonzero_indices, image_dims)
    plot_peak_spatial_uncertainty!(paths, results, x_true, nonzero_indices, image_dims)

    return paths
end

function benchmark_snr(A, x_true, step, snr)
    y = load_single("y_noisy_SNR$(snr).jld2")
    noise = load_single("L2_norm_of_noise_SNR$(snr).jld2")
    rng = MersenneTwister(10_000 + snr)

    println("SNR $(snr)")
    @printf("  noise=%9.4g target_residual=%9.4g\n", noise, residual(A, y, x_true))
    best, search_seconds = solve_noise_matched_lasso(A, y, noise, step)
    x_hat = best.info.x
    @printf(
        "  selected lambda=%9.3g residual=%9.4g residual-noise=%9.3g rel_x_err=%9.4g search_total=%7.3f s\n",
        best.lambda,
        best.residual,
        best.gap,
        relative_error(x_hat, x_true),
        search_seconds,
    )

    common_kwargs = (
        algorithm=:fista,
        step=step,
        abstol=SOLVER_ABSTOL,
        reltol=SOLVER_RELTOL,
        maxiter=FISTA_MAXITER,
        selection_tol=SELECTION_TOL,
    )

    bootstrap, bootstrap_seconds = timed(
        () -> nonnegative_lasso_bootstrap_uncertainty(
            A,
            y,
            best.lambda;
            samples=UNCERTAINTY_SAMPLES,
            rng=rng,
            common_kwargs...,
        ),
    )
    noise_uncertainty, noise_seconds = timed(
        () -> nonnegative_lasso_noise_perturbation_uncertainty(
            A,
            y,
            best.lambda;
            samples=UNCERTAINTY_SAMPLES,
            noise_norm=noise,
            rng=rng,
            common_kwargs...,
        ),
    )
    stability, stability_seconds = timed(
        () -> nonnegative_lasso_stability_selection(
            A,
            y,
            best.lambda;
            samples=UNCERTAINTY_SAMPLES,
            rng=rng,
            threshold=0.8,
            common_kwargs...,
        ),
    )
    lambdas = exp.(range(log(max(best.lambda / 4, eps())), log(best.lambda * 4); length=LAMBDA_PATH_STEPS))
    path, path_seconds = timed(
        () -> nonnegative_lasso_lambda_path(
            A,
            y,
            reverse(lambdas);
            common_kwargs...,
        ),
    )
    refit, refit_seconds = timed(
        () -> lasso_refit(
            A,
            y,
            x_hat;
            positive=true,
            noise_variance=noise_sigma(noise, y)^2,
            support_tol=SELECTION_TOL,
        ),
    )


    rows = []
    push!(
        rows,
        (
            snr=snr,
            estimator="fista",
            seconds=search_seconds,
            lambda=best.lambda,
            residual=best.residual,
            residual_minus_noise=best.gap,
            relative_x_error=relative_error(x_hat, x_true),
            active_selection_mean=NaN,
            inactive_selection_mean=NaN,
            interval_coverage=NaN,
            active_interval_coverage=NaN,
            inactive_interval_coverage=NaN,
        ),
    )

    for (name, uncertainty, seconds) in (
        ("bootstrap", bootstrap, bootstrap_seconds),
        ("noise", noise_uncertainty, noise_seconds),
    )
        coverage, active_coverage, inactive_coverage = summarize_interval(uncertainty, x_true)
        selection = summarize_selection(uncertainty.selection_probability, x_true)
        push!(
            rows,
            (
                snr=snr,
                estimator=name,
                seconds=seconds,
                lambda=best.lambda,
                residual=residual(A, y, uncertainty.coefficient_mean),
                residual_minus_noise=residual(A, y, uncertainty.coefficient_mean) - noise,
                relative_x_error=relative_error(uncertainty.coefficient_mean, x_true),
                active_selection_mean=selection.active_mean,
                inactive_selection_mean=selection.inactive_mean,
                interval_coverage=coverage,
                active_interval_coverage=active_coverage,
                inactive_interval_coverage=inactive_coverage,
            ),
        )
        @printf(
            "  %-10s time=%7.3f s rel_mean_err=%9.4g active_sel=%6.3f inactive_sel=%6.3f coverage=%6.3f\n",
            name,
            seconds,
            relative_error(uncertainty.coefficient_mean, x_true),
            selection.active_mean,
            selection.inactive_mean,
            coverage,
        )
    end

    stability_selection = summarize_selection(vec(stability.selection_probability), x_true)
    push!(
        rows,
        (
            snr=snr,
            estimator="stability",
            seconds=stability_seconds,
            lambda=best.lambda,
            residual=NaN,
            residual_minus_noise=NaN,
            relative_x_error=NaN,
            active_selection_mean=stability_selection.active_mean,
            inactive_selection_mean=stability_selection.inactive_mean,
            interval_coverage=NaN,
            active_interval_coverage=NaN,
            inactive_interval_coverage=NaN,
        ),
    )
    @printf(
        "  %-10s time=%7.3f s active_sel=%6.3f inactive_sel=%6.3f selected80=%d\n",
        "stability",
        stability_seconds,
        stability_selection.active_mean,
        stability_selection.inactive_mean,
        stability_selection.selected_at_80,
    )

    refit_residual = residual(A, y, refit.x)
    push!(
        rows,
        (
            snr=snr,
            estimator="refit",
            seconds=refit_seconds,
            lambda=best.lambda,
            residual=refit_residual,
            residual_minus_noise=refit_residual - noise,
            relative_x_error=relative_error(refit.x, x_true),
            active_selection_mean=NaN,
            inactive_selection_mean=NaN,
            interval_coverage=NaN,
            active_interval_coverage=NaN,
            inactive_interval_coverage=NaN,
        ),
    )
    @printf(
        "  %-10s time=%7.3f s support=%d original_support=%d residual=%9.4g rel_x_err=%9.4g\n",
        "refit",
        refit_seconds,
        length(refit.support),
        length(refit.original_support),
        refit_residual,
        relative_error(refit.x, x_true),
    )



    closest_path_index = argmin(abs.(path.residuals .- noise))
    @printf(
        "  %-10s time=%7.3f s closest_lambda=%9.3g closest_residual=%9.4g active=%d\n",
        "lambda_path",
        path_seconds,
        path.lambdas[closest_path_index],
        path.residuals[closest_path_index],
        count(path.active[:, closest_path_index]),
    )

    return (
        snr=snr,
        y=y,
        noise_norm=noise,
        lambda=best.lambda,
        x_hat=x_hat,
        bootstrap=bootstrap,
        noise=noise_uncertainty,
        stability=stability,
        path=path,
        refit=refit,
        rows=rows,
    )
end

function main()
    BLAS.set_num_threads(BLAS_THREADS)
    A = load_single("A.jld2")
    target = load_single("target.jld2")
    image_dims = size(target)
    x_true = vec(permutedims(target))
    nonzero_indices = nonzero_peak_indices(x_true)
    step, step_seconds = timed(() -> fista_step_size(A; method=:power))

    println("Test-data nonnegative LASSO uncertainty benchmark")
    println("A_size=$(size(A)), BLAS_threads=$(BLAS.get_num_threads())")
    println("snrs=$(SNRS), uncertainty_samples=$(UNCERTAINTY_SAMPLES), lambda_path_steps=$(LAMBDA_PATH_STEPS)")
    println("target_nonzero_count=$(length(nonzero_indices))")
    println("FISTA_step=$(step) computed in $(round(step_seconds; digits=3)) s")
    println("maxiter=$(FISTA_MAXITER), abstol=$(SOLVER_ABSTOL), reltol=$(SOLVER_RELTOL)")
    println()

    results = [benchmark_snr(A, x_true, step, snr) for snr in SNRS]
    rows = reduce(vcat, [collect(result.rows) for result in results])
    table_path = write_summary_table(rows)
    raw_path = write_raw_results(results, rows, x_true, nonzero_indices, image_dims)
    plot_paths = maybe_plot_results(results, rows, x_true, image_dims, nonzero_indices)

    println()
    println("Wrote summary table: $(table_path)")
    println("Wrote raw uncertainty results: $(raw_path)")
    if isempty(plot_paths)
        println("No plots written. Install CairoMakie in the project to enable PNG output.")
    else
        println("Wrote plots:")
        foreach(path -> println("  $(path)"), plot_paths)
    end
end

main()
