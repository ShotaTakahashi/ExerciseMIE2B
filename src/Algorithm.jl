module Algorithm
    using LinearAlgebra
    using CSV, DataFrames

    include("Operator.jl")

    mutable struct IterativeAlgorithm
        xk::Vector{Float64}
        xk_1::Vector{Float64}
        maximum_iter::Int64
        tol::Float64

        IterativeAlgorithm(x0, maximum_iter, tol) = new(x0, x0, maximum_iter, tol)
    end

    struct Param
        stepsize::Float64
        theta::Float64
    end

    function run(alg::IterativeAlgorithm, obj, grad, update, param::Param, result_dir="")
        is_write = false
        if result_dir !== ""
            is_write = true
            file = open(result_dir, "w+")
            close(file)
            df = DataFrame(:iter => [0], :obj => obj(alg.xk))
        end
        for i = 1:alg.maximum_iter
            alg.xk_1, alg.xk = alg.xk, update(alg.xk, grad, param)
            if is_write
                push!(df, (i, obj(alg.xk)))
            end
            if stop_criteria(alg.xk_1, alg.xk, alg.tol)
                if is_write
                    CSV.write(result_dir, df)
                end
                return alg.xk
            end
        end
        if is_write
            CSV.write(result_dir, df)
        end
        return alg.xk
    end

    function update_ISTA(x::Vector{Float64}, grad, param::Param)
        grad = grad(x)
        v = x - grad .* param.stepsize
        v = Operator.soft_thresholding(v, param.theta)
        return v
    end

    run_ISTA(alg, obj, grad, param, result_dir) = run(alg, obj, grad, update_ISTA, param, result_dir)

    stop_criteria(x, y, tol) = norm(x - y, 2) < tol
end