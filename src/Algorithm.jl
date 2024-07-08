module Algorithm
    using LinearAlgebra
    using CSV, DataFrames

    include("Operator.jl")

    mutable struct IterativeAlgorithm
        xk::Vector{Float64}
        xk_old::Vector{Float64}
        maximum_iter::Int64
        tol::Float64
        df::DataFrame

        IterativeAlgorithm(x0, maximum_iter, tol) = new(x0, x0, maximum_iter, tol, DataFrame([Int64[], Float64[], Float64[]], ["iter", "time", "obj"]))
    end

    function run(alg::IterativeAlgorithm, obj, grad, update, stepsize::Float64, theta::Float64)
        push!(alg.df, (0, 0.0, obj(alg.xk)))
        time = 0.0
        for i = 1:alg.maximum_iter
            time += @elapsed begin
                update(alg, grad, stepsize, theta)
                if stop_criteria(alg.xk_old, alg.xk, alg.tol)
                    push!(alg.df, (i, time, obj(alg.xk)))
                    break
                end
                push!(alg.df, (i, time, obj(alg.xk)))
            end
        end
        return alg.xk
    end

    function update_ISTA(alg::IterativeAlgorithm, grad, stepsize::Float64, theta::Float64)
        alg.xk_old, alg.xk = alg.xk, prox(alg.xk, grad, stepsize, theta)
    end

    # hoge は適切な名称に変えてください．
    function update_hoge(alg::IterativeAlgorithm, grad, stepsize::Float64, theta::Float64)
        alg.xk_old, alg.xk = alg.xk, prox(alg.xk, grad, stepsize, theta)
    end

    function prox(x::Vector{Float64}, grad, stepsize::Float64, theta::Float64)
        grad = grad(x)
        v = x - grad .* stepsize
        v = Operator.soft_thresholding(v, theta * stepsize)
        return v
    end

    run_ISTA(alg, obj, grad, stepsize, theta) = run(alg, obj, grad, update_ISTA, stepsize, theta)

    # hoge は適切な名称に変えてください．
    run_hoge(alg, obj, grad, stepsize, theta) = run(alg, obj, grad, update_hoge, stepsize, theta)

    stop_criteria(x, y, tol) = norm(x - y, 2) < tol
end