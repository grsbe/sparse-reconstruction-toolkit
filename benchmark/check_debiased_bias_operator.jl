using JLD2
using LinearAlgebra
using Printf
using SparseReconstructionToolkit

const DATA_DIR = joinpath(@__DIR__, "..", "testdata")
load_single(name) = JLD2.load(joinpath(DATA_DIR, name))["single_stored_object"]

function main()
    BLAS.set_num_threads(32)
    A = load_single("A.jld2")
    y = load_single("y_noisy_SNR30.jld2")
    noise = load_single("L2_norm_of_noise_SNR30.jld2")
    lambda = 2.1266246784487773e-7
    step = fista_step_size(A; method=:power)
    info = nonnegative_lasso_fista(
        LassoFISTASolver(A, y; step),
        lambda;
        abstol=1e-7,
        reltol=1e-5,
        maxiter=2000,
        return_info=true,
    )

    m, n = size(A)
    gram = transpose(A) * A
    empirical = gram ./ m
    s = svdvals(empirical)
    tol_default = max(size(empirical)...) * eps(eltype(empirical)) * maximum(s)
    positive_s = s[s .> tol_default]

    @printf("A size = %s, m=%d, n=%d\n", string(size(A)), m, n)
    @printf("lambda = %.16g, noise = %.16g\n", lambda, noise)
    @printf("FISTA residual = %.16g, converged = %s, iterations = %d\n", norm(A * info.x - y), string(info.converged), info.iterations)
    @printf("svd max = %.6e, min = %.6e, default tol = %.6e\n", maximum(s), minimum(s), tol_default)
    @printf("numerical rank = %d, nullity = %d\n", length(positive_s), n - length(positive_s))
    @printf("smallest positive sv = %.6e, cond positive = %.6e\n", minimum(positive_s), maximum(positive_s) / minimum(positive_s))

    M = pinv(empirical)
    bias = I - M * empirical
    @printf("norm(I - pinv(G)G, Inf) = %.16g\n", norm(bias, Inf))
    @printf("norm(I - G pinv(G), Inf) = %.16g\n", norm(I - empirical * M, Inf))
    @printf("norm(G pinv(G) G - G) / norm(G) = %.16g\n", norm(empirical * M * empirical - empirical) / norm(empirical))

    debiased = debiased_lasso(A, y, lambda; x_hat=info.x, positive=true, noise_norm=noise)
    @printf("debiased_lasso bias_operator_infnorm = %.16g\n", debiased.bias_operator_infnorm)
end

main()
