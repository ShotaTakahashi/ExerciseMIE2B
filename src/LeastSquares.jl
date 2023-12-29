using CSV
using DataFrames
using LinearAlgebra
using Random
using SparseArrays
using Plots

include("Algorithm.jl")
include("Operator.jl")

loss(x) = norm(A*x - b)^2/2
reg(x) = theta * sum(abs.(x))
obj(x) = loss(x) + reg(x)

grad(x) = A' * (A *x - b)

rng = MersenneTwister(1234)
n = 200
m = 300 # 自由に変えてもよいですが，n < m を推奨．
sparsity = 0.05
theta = .00015 # θ とすることも可能

A = randn(rng, Float64, (m, n))
foreach(normalize!, eachcol(A))
x = Array(sprandn(rng, n, sparsity))
b = A * x

lambda = 1.0

rng = MersenneTwister(5)
x0 = randn(rng, Float64, n)
maximum_iter = 10
tol = 1e-1
alg = Algorithm.IterativeAlgorithm(x0, maximum_iter, tol)
param = Algorithm.Param(lambda, theta)

if !ispath("./results")
    mkdir("./results")
end
result_dir = "./results/least-squares_$n-$m-$sparsity-$theta-$lambda"
@time Algorithm.run_ISTA(alg, obj, grad, param)
CSV.write(result_dir * ".csv", alg.df)

df = alg.df
obj_opt = obj(x)
plot(df.iter, abs.(df.obj .- obj_opt), xaxis="k", yaxis=raw"$\log_{10}|Ψ(x^k) - Ψ(x^*)|$", label="PG", yscale=:log10)
savefig(result_dir * "-diffobj.png")
plot(df.iter, df.obj, xaxis="k", yaxis=raw"$\log_{10}|Ψ(x^k)|$", label="PG", yscale=:log10)
savefig(result_dir * "-objv.png")
