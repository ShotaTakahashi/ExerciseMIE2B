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

        IterativeAlgorithm(x0, maximum_iter, tol) = new(x0, x0, maximum_iter, tol, DataFrame([Int64[], Float64[]], ["iter", "obj"]))
    end

    struct Param
        stepsize::Float64
        theta::Float64
    end

    function run(alg::IterativeAlgorithm, obj, grad, update, param::Param)
        push!(alg.df, (0, obj(alg.xk)))
        for i = 1:alg.maximum_iter
            alg.xk_old, alg.xk = alg.xk, update(alg.xk, grad, param)
            push!(alg.df, (i, obj(alg.xk)))
            if stop_criteria(alg.xk_old, alg.xk, alg.tol)
                return alg.xk
            end
        end
        return alg.xk
    end

    function update_ISTA(x::Vector{Float64}, grad, param::Param)
        grad = grad(x)
        v = x - grad .* param.stepsize
        v = Operator.soft_thresholding(v, param.theta * param.stepsize)
        return v
    end

    run_ISTA(alg, obj, grad, param) = run(alg, obj, grad, update_ISTA, param)

    stop_criteria(x, y, tol) = norm(x - y, 2) < tol
end