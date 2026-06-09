using CairoMakie
using JLD2
using LinearAlgebra
using Printf
using Statistics
using SparseReconstructionToolkit

const DATA_DIR = joinpath(@__DIR__, "..", "testdata")
const PLOT_DIR = joinpath(@__DIR__, "..", "benchmarkplots")
const MARGINS = (0, 2, 5, 10, 15, 20)

load_single(name) = JLD2.load(joinpath(DATA_DIR, name))["single_stored_object"]

function timed(f)
    t0 = time_ns()
    result = f()
    return result, (time_ns() - t0) / 1e9
end

function vector_to_map(x, image_dims)
    return permutedims(reshape(x, reverse(image_dims)))
end

function plot_map!(figpos, image, title)
    ax = Axis(figpos, title=title, aspect=DataAspect())
    hidedecorations!(ax)
    heatmap!(ax, image[:, reverse(axes(image, 2))]; colormap=:viridis)
    return ax
end

function center_mask(image_dims, margin)
    mask = trues(image_dims)
    if margin > 0
        mask[1:margin, :] .= false
        mask[end-margin+1:end, :] .= false
        mask[:, 1:margin] .= false
        mask[:, end-margin+1:end] .= false
    end
    return mask
end

function mask_to_vector(mask)
    return vec(permutedims(mask))
end

function write_spatial_table(rows)
    mkpath(PLOT_DIR)
    path = joinpath(PLOT_DIR, "testdata_debiased_bias_spatial_summary.tsv")
    open(path, "w") do io
        println(io, join(("margin", "center_pixels", "outer_pixels", "center_rows_full_infnorm", "outer_rows_full_infnorm", "center_block_infnorm", "outer_block_infnorm", "center_diag_abs_max", "outer_diag_abs_max", "center_row_sum_median", "outer_row_sum_median"), "\t"))
        for row in rows
            println(io, join(row, "\t"))
        end
    end
    return path
end

function main()
    BLAS.set_num_threads(32)
    mkpath(PLOT_DIR)

    A = load_single("A.jld2")
    target = load_single("target.jld2")
    image_dims = size(target)
    m, n = size(A)
    @printf("A size=%s image_dims=%s BLAS_threads=%d\n", string(size(A)), string(image_dims), BLAS.get_num_threads())

    gram, gram_seconds = timed(() -> transpose(A) * A ./ m)
    @printf("Gram computed in %.3f s\n", gram_seconds)
    M, pinv_seconds = timed(() -> pinv(gram))
    @printf("pinv computed in %.3f s\n", pinv_seconds)
    bias, bias_seconds = timed(() -> Matrix{Float64}(I, n, n) - M * gram)
    @printf("bias operator computed in %.3f s\n", bias_seconds)

    row_sums = vec(sum(abs, bias; dims=2))
    diag_abs = abs.(diag(bias))
    signed_diag = diag(bias)
    @printf("global infnorm=%.16g median row sum=%.6g max diag abs=%.6g\n", maximum(row_sums), median(row_sums), maximum(diag_abs))

    rows = []
    for margin in MARGINS
        mask_map = center_mask(image_dims, margin)
        mask = mask_to_vector(mask_map)
        outer = .!mask
        center_rows_full = any(mask) ? maximum(row_sums[mask]) : NaN
        outer_rows_full = any(outer) ? maximum(row_sums[outer]) : NaN
        center_block = any(mask) ? norm(bias[mask, mask], Inf) : NaN
        outer_block = any(outer) ? norm(bias[outer, outer], Inf) : NaN
        center_diag_max = any(mask) ? maximum(diag_abs[mask]) : NaN
        outer_diag_max = any(outer) ? maximum(diag_abs[outer]) : NaN
        center_median = any(mask) ? median(row_sums[mask]) : NaN
        outer_median = any(outer) ? median(row_sums[outer]) : NaN
        push!(rows, (margin, count(mask), count(outer), center_rows_full, outer_rows_full, center_block, outer_block, center_diag_max, outer_diag_max, center_median, outer_median))
        @printf("margin=%2d center=%4d outer=%4d center_rows_full=%.6g outer_rows_full=%.6g center_block=%.6g outer_block=%.6g center_diag_max=%.6g outer_diag_max=%.6g\n", margin, count(mask), count(outer), center_rows_full, outer_rows_full, center_block, outer_block, center_diag_max, outer_diag_max)
    end

    table_path = write_spatial_table(rows)

    row_sum_map = vector_to_map(row_sums, image_dims)
    diag_abs_map = vector_to_map(diag_abs, image_dims)
    signed_diag_map = vector_to_map(signed_diag, image_dims)

    fig = Figure(size=(1250, 420))
    plot_map!(fig[1, 1], row_sum_map, "row abs-sum of I - M G")
    plot_map!(fig[1, 2], diag_abs_map, "abs diag of I - M G")
    plot_map!(fig[1, 3], signed_diag_map, "diag of I - M G")
    map_path = joinpath(PLOT_DIR, "testdata_debiased_bias_spatial_maps.png")
    save(map_path, fig)

    margins = [row[1] for row in rows]
    center_rows = [row[4] for row in rows]
    outer_rows = [row[5] for row in rows]
    center_blocks = [row[6] for row in rows]
    outer_blocks = [row[7] for row in rows]
    fig = Figure(size=(900, 420))
    ax = Axis(fig[1, 1], xlabel="excluded border margin (pixels)", ylabel="Inf norm", yscale=log10)
    lines!(ax, margins, center_rows; label="center rows, full cols", linewidth=2)
    scatter!(ax, margins, center_rows)
    lines!(ax, margins, outer_rows; label="outer rows, full cols", linewidth=2)
    scatter!(ax, margins, outer_rows)
    lines!(ax, margins, center_blocks; label="center block", linewidth=2, linestyle=:dash)
    scatter!(ax, margins, center_blocks)
    lines!(ax, margins, outer_blocks; label="outer block", linewidth=2, linestyle=:dash)
    scatter!(ax, margins, outer_blocks)
    axislegend(ax, position=:rt)
    curve_path = joinpath(PLOT_DIR, "testdata_debiased_bias_spatial_margins.png")
    save(curve_path, fig)

    println("Wrote spatial table: $(table_path)")
    println("Wrote spatial maps: $(map_path)")
    println("Wrote margin curves: $(curve_path)")
end

main()
