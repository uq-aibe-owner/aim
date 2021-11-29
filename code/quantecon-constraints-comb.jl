
#=using Pkg
import Pkg; Pkg.add("MathOptInterface")
import Pkg; Pkg.add("Statistics")
import Pkg; Pkg.add("QuantEcon")
import Pkg; Pkg.add("Interpolations")
import Pkg; Pkg.add("NLsolve")
import Pkg; Pkg.add("Optim")
import Pkg; Pkg.add("Random")
import Pkg; Pkg.add("IterTools")
import Pkg; Pkg.add("JuMP")
import Pkg; Pkg.add("Ipopt")
import Pkg; Pkg.add("BenchmarkTools")
import Pkg; Pkg.add("GaussianProcesses")
=#

using
    MathOptInterface,
    BenchmarkTools,
    LinearAlgebra,
    Statistics,
    QuantEcon,
    Interpolations,
    NLsolve,
    Optim,
    Random,
    IterTools,
    JuMP,
    Ipopt,
    GaussianProcesses,
    TimerOutputs,
    Random;


δ = 0.1
β = 0.86
ϕ = 0.6
ξ = 0.5
μ = 0.5
ψ = 1/ϕ
b = 1/(ϕ^ϕ * (1-ϕ)^(1-ϕ))


numSec = 2
numLin = 10
gridMin = 2
gridMax = 30
gridSiz = numLin^numSec
u(x,y) = (log(x) + log(y))
wInit(x) = (log(x[1]) + log(x[2]))/(1 - β)
function gradwInit(g, x, y)
    g[1] = 1/x/(1-β)
    g[2] = 1/y/(1-β)
end
function hesswInit(h, x, y)
    h[1,1] = -1/x^2/(1-β)
    h[1,2] = 0
    h[2,1] = 0
    h[2,2] = -1/y^2/(1-β)
end
yTargCbd = Vector{Float64}[]
sTargCbd = Vector{Float64}[]
mTargCbd = Matrix{Float64}[]
xTargCbd = Matrix{Float64}[]
for i in 1:gridSiz
    push!(yTargCbd, zeros(numSec))
    push!(sTargCbd, zeros(numSec))
    push!(mTargCbd, zeros(numSec,numSec))
    push!(xTargCbd, zeros(numSec, numSec))
end
wTargCbd = Float64[]
#randvec = 20 .+ 30*rand(rng,numLin)
#for p in product(randvec, randvec)
#    push!(grid, collect(p))
#end
grid = Vector{Float64}[]
for p in product(LinRange(gridMin, gridMax, numLin),LinRange(gridMin, gridMax, numLin))
    push!(grid, collect(p))
end
rng = MersenneTwister(1234)
#for timing
tOCbd = TimerOutput()
#wval = w.(grid)
#wval = w.(grid)
    
#the following function generates target values (of the value function) for the ML process
##note that (unlike Sargent and Stachursky) the grid is written in terms of kapital
#this simplified the problem substantially as otherwise intermediate variables became dynamic
##also note that this accepts the smooth function
function TargCbd(w, grid, β ; return_policy= false)
    @timeit tOCbd string("Combined-for-", length(grid),"-pts") begin
    modTrial = Model(Ipopt.Optimizer)
    set_silent(modTrial)
    @variable(modTrial,  c[1:numSec, 1:length(grid)] >= 0.0001)
    @variable(modTrial, y[1:numSec, 1:length(grid)] >= 0.0001)
    @variable(modTrial, kp[1:numSec, 1:length(grid)]>= 0.0001)
    @variable(modTrial, x[1:numSec, 1:numSec, 1:length(grid)]>=0.0001)
    @variable(modTrial, xComb[1:numSec, 1:length(grid)]>=0.0001)
    @variable(modTrial, m[1:numSec, 1:numSec, 1:length(grid)]>=0.0001)
    @variable(modTrial, mComb[1:numSec, 1:length(grid)]>=0.0001)
    @variable(modTrial, u[1:length(grid)])
    @variable(modTrial, w[1:length(grid)])
    k = grid
    for n in 1:length(grid)
        for i in 1:numSec
            #output
            @NLconstraint(modTrial, y[i,n] == b * k[n][i]^ϕ * mComb[i,n]^(1-ϕ))
            #future capital
            @constraint(modTrial, kp[i,n] == xComb[i,n] + (1-δ) * k[n][i])
            #supply equals demand (eq'm constraint)
            @constraint(modTrial, 0 == y[i,n] - c[i,n] - sum(m[i,:,n]) - sum(x[i,:,n]))
        end
        @NLconstraint(modTrial, xComb[1,n] == x[1,1,n]^ξ * x[2,1,n]^(1-ξ))
        @NLconstraint(modTrial, xComb[2,n] == x[1,2,n]^ξ * x[2,2,n]^(1-ξ))
        @NLconstraint(modTrial, mComb[1,n] == m[1,1,n]^μ * m[2,1,n]^(1-μ))
        @NLconstraint(modTrial, mComb[2,n] == m[1,2,n]^μ * m[2,2,n]^(1-μ))
        @NLconstraint(modTrial, u[n] == log(c[1,n]) + log(c[2,n]))
        @NLconstraint(modTrial, w[n] == (log(kp[1,n]) + log(kp[2,n]))/(1-β))
#=             register(modTrial, :wInit, 2, wInit, gradwInit)
            @NLconstraint(modTrial, w == wInit(kp[1],kp[2]))
            register(modTrial, :u, 2, u; autodiff = true) =#
    end
    @NLobjective(modTrial, Max, sum(u[i] + β * w[i] for i in 1:length(grid)))
    JuMP.optimize!(modTrial)
    #push!(wTargCbd, value.(w))
    if return_policy
        for i in 1:length(grid)
        sTargCbd[i] = value.(c)[1:2, i]
        yTargCbd[i] = value.(y)[1:2, i]
        mTargCbd[i] = value.(m)[1:2,1:2, i]
        xTargCbd[i] = value.(x)[1:2,1:2, i]
        end
    end
                   end #of timeit
    #global intK[n] = value.(f)
    #global prevK = intK
    return value.(w)
end

function evalBellCbd()
    wTargCbd = TargCbd(wInit, grid, β; return_policy=false)
    show(tOCbd)
    println("\n ---------")
    return wTargCbd
end
#=XTargCbd = Vector{Float64}[]
       for j = 1:numLin^numSec
           push!(XTargCbd, sum(xTargCbd[j][:,i] for i in 1:numSec))
       end
MTargCbd = Vector{Float64}[]
       for j = 1:numLin^numSec
           push!(MTargCbd, sum(mTargCbd[j][:,i] for i in 1:numSec))
       end
fTargCbd = sTargCbd + MTargCbd + XTargCbd

MPCf = Vector{Float64}[]
       for j = 1:numLin^numSec
           push!(MPCf, sTargCbd[j]./fTargCbd[j])
       end
MPC = Vector{Float64}[]
       for j = 1:numLin^numSec
           push!(MPC, sTargCbd[j]./yTargCbd[j])
       end
MPM = Vector{Float64}[]
       for j = 1:numLin^numSec
           push!(MPM, MTargCbd[j]./yTargCbd[j])
       end
MPX = Vector{Float64}[]
       for j = 1:numLin^numSec
           push!(MPX, XTargCbd[j]./yTargCbd[j])
       end
# a function for turning vectors of vectors into matrices
#function vvm(x)
#           dim1 = length(x)
#           dim2 = length(x[1])
#           matrix = zeros(Float64, dim1, dim2)
#           @inbounds @fastmath for i in 1:dim1, for j in 1:dim2
#                   matrix[i, j] = x[i][j]
#               end
#           end
#           return matrix
#end
#alternatively
gridM = reduce(hcat, grid)

mz = MeanZero()
ml = MeanLin([2., 3.])
mp = MeanPoly([2. 3.; 1. 2.])
kern = SE(0.0, 0.0)
gp = GP(gridM, wTargCbd, mz, kern)
#gp = GP(gridM, wTargCbd, MeanLin([2., 3.]), kern)
GaussianProcesses.optimize!(gp)
predict_f(gp, gridM)[1]
fstar(x) = predict_f(gp,hcat(x))[1]

plot!(contour(gp), heatmap(gp); fmt=:png)
#wTargp(x, y) = predict_f(gp,hcat([x, y]))[1]=#
