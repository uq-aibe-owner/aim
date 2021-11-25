
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
    Random;


δ = 0.1
β = 0.86
ϕ = 0.6
ξ = 0.5
μ = 0.5
ψ = 1/ϕ
b = 1/(ϕ^ϕ * (1-ϕ)^(1-ϕ))


numSectors = 2
numPoints1D = 500
gridMax = 50
gridMin = 10

u(x,y) = (log(x) + log(y))
wInit(x,y) = (log(x) + log(y))/(1 - β)
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
grid = Vector{Float64}[]
yTarg = Vector{Float64}[]
sTarg = Vector{Float64}[]
mTarg = Matrix{Float64}[]
xTarg = Matrix{Float64}[]
wTarg = Float64[]
rng = MersenneTwister(1234)
for p in product(LinRange(gridMin,gridMax,numPoints1D), LinRange(gridMin, gridMax,numPoints1D))
    push!(grid, collect(p))
end
#wval = w.(grid)
    
#the following function generates target values (of the value function) for the ML process
##note that (unlike Sargent and Stachursky) the grid is written in terms of kapital
#this simplified the problem substantially as otherwise intermediate variables became dynamic
##also note that this accepts the smooth function
function Targ(w, grid, β ; compute_policy = false)

    for n in 1:length(grid)
        k = grid[n]
        modTrial = Model(Ipopt.Optimizer)
        set_silent(modTrial)
        @variable(modTrial,  c[1:numSectors] >= 0.0001)
        @variable(modTrial, y[1:numSectors] >= 0.0001)
        @variable(modTrial, kp[1:numSectors]>= 0.0001)
        @variable(modTrial, x[1:numSectors, 1:numSectors]>=0.0001)
        @variable(modTrial, xComb[1:numSectors]>=0.0001)
        @variable(modTrial, m[1:numSectors, 1:numSectors]>=0.0001)
        @variable(modTrial, mComb[1:numSectors]>=0.0001)
        @variable(modTrial, u)
        @variable(modTrial, w)
        for i in 1:numSectors
            #output
            @NLconstraint(modTrial, y[i] == b * k[i]^ϕ * mComb[i]^(1-ϕ))
            #future capital
            @constraint(modTrial, kp[i] == xComb[i] + (1-δ) * k[i])
            #supply equals demand (eq'm constraint)
            @constraint(modTrial, 0 == y[i] - c[i] - sum(m[i,:]) - sum(x[i,:]))
        end
        @NLconstraint(modTrial, xComb[1] == x[1,1]^ξ * x[2,1]^(1-ξ))
        @NLconstraint(modTrial, xComb[2] == x[1,2]^ξ * x[2,2]^(1-ξ))
        @NLconstraint(modTrial, mComb[1] == m[1,1]^μ * m[2,1]^(1-μ))
        @NLconstraint(modTrial, mComb[2] == m[1,2]^μ * m[2,2]^(1-μ))
        @NLconstraint(modTrial, u == log(c[1]) + log(c[2]))
        @NLconstraint(modTrial, w == (log(kp[1]) + log(kp[2]))/(1-β))
#        register(modTrial, :wInit, 2, wInit, gradwInit)
#        @NLconstraint(modTrial, w == wInit(kp[1],kp[2]))
#        register(modTrial, :u, 2, u; autodiff = true)
        @NLobjective(modTrial, Max, u + β * w)
        JuMP.optimize!(modTrial)
        push!(wTarg, JuMP.objective_value(modTrial))
        if compute_policy
           push!(xTarg, value.(x))
           push!(sTarg, value.(c))
           push!(mTarg, value.(m))
           push!(yTarg, value.(y))
        end
#        global intK[n] = value.(f)
    end
    #global prevK = intK
    if compute_policy
        return wTarg
    end
    return
end

function evalBell()
 wTargC  = Targ(wInit, grid, β; compute_policy=true)
    return wTargC
end
#=
XTarg = Vector{Float64}[]
       for j = 1:numPoints1D^numSectors
           push!(XTarg, sum(xTarg[j][:,i] for i in 1:numSectors))
       end
MTarg = Vector{Float64}[]
       for j = 1:numPoints1D^numSectors
           push!(MTarg, sum(mTarg[j][:,i] for i in 1:numSectors))
       end
fTarg = sTarg + MTarg + XTarg

MPCf = Vector{Float64}[]
       for j = 1:numPoints1D^numSectors
           push!(MPCf, sTarg[j]./fTarg[j])
       end
MPC = Vector{Float64}[]
       for j = 1:numPoints1D^numSectors
           push!(MPC, sTarg[j]./yTarg[j])
       end
MPM = Vector{Float64}[]
       for j = 1:numPoints1D^numSectors
           push!(MPM, MTarg[j]./yTarg[j])
       end
MPX = Vector{Float64}[]
       for j = 1:numPoints1D^numSectors
           push!(MPX, XTarg[j]./yTarg[j])
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
kern = SE(log(2), log(2000.0))
gp = GP(gridM, wTarg+0.2*randn(numPoints1D^numSectors), mz, kern)
#gp = GP(gridM, wTarg, MeanLin([2., 3.]), kern)
GaussianProcesses.optimize!(gp)
predict_f(gp, gridM)[1]
fstar(x) = predict_f(gp,hcat(x))[1]

plot!(contour(gp), heatmap(gp); fmt=:png)
#wTargp(x, y) = predict_f(gp,hcat([x, y]))[1]
=#
