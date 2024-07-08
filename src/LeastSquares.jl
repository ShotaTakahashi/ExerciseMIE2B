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

result_dir = "./results/least-squares_$n-$m-$sparsity-$theta"
if !ispath("./results")
    mkdir("./results")
end

pg = Algorithm.IterativeAlgorithm(x0, maximum_iter, tol)
@time Algorithm.run_ISTA(pg, obj, grad, lambda, theta)
CSV.write(result_dir * "-pg.csv", pg.df)

# アルゴリズムを加える場合は Algorithm.jl に run_hoge を書き直しましょう．"hoge" や "HOGE" は適宜変更してください．
hoge = Algorithm.IterativeAlgorithm(x0, maximum_iter, tol)
@time Algorithm.run_hoge(hoge, obj, grad, lambda, theta)
CSV.write(result_dir * "-hoge.csv", hoge.df)

df_pg = pg.df
df_hoge = hoge.df
obj_opt = obj(x)
plot(df_pg.iter, abs.(df_pg.obj .- obj_opt), xaxis="k", yaxis=raw"$\log_{10}|Ψ(x^k) - Ψ(x^*)|$", label="PG", yscale=:log10)
plot!(df_hoge.iter, abs.(df_hoge.obj .- obj_opt), xaxis="k", yaxis=raw"$\log_{10}|Ψ(x^k) - Ψ(x^*)|$", label="HOGE", yscale=:log10)
savefig(result_dir * "-diffobj.png")
print("saved ", result_dir * "-diffobj.png\n")
plot(df_pg.iter, df_pg.obj, xaxis="k", yaxis=raw"$\log_{10}|Ψ(x^k)|$", label="PG", yscale=:log10)
plot!(df_hoge.iter, df_hoge.obj, xaxis="k", yaxis=raw"$\log_{10}|Ψ(x^k)|$", label="HOGE", yscale=:log10)
savefig(result_dir * "-objv.png")
print("saved ", result_dir * "-objv.png\n")

plot(df_pg.time, abs.(df_pg.obj .- obj_opt), xaxis="time (sec)", yaxis=raw"$\log_{10}|Ψ(x^k) - Ψ(x^*)|$", label="PG", yscale=:log10)
plot!(df_hoge.time, abs.(df_hoge.obj .- obj_opt), xaxis="time (sec)", yaxis=raw"$\log_{10}|Ψ(x^k) - Ψ(x^*)|$", label="HOGE", yscale=:log10)
savefig(result_dir * "-diffobj-time.png")
print("saved ", result_dir * "-diffobj-time.png\n")
plot(df_pg.time, df_pg.obj, xaxis="time (sec)", yaxis=raw"$\log_{10}|Ψ(x^k)|$", label="PG", yscale=:log10)
plot!(df_hoge.time, df_hoge.obj, xaxis="time (sec)", yaxis=raw"$\log_{10}|Ψ(x^k)|$", label="HOGE", yscale=:log10)
savefig(result_dir * "-objv-time.png")
print("saved ", result_dir * "-objv-time.png\n")
