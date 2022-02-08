
#= using Pkg
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
import Pkg; Pkg.add("TimerOutputs") =#

using MathOptInterface,
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
    # GaussianProcesses,
    Random,
    TimerOutputs;
    # DateTime;

γ = 0.6
σ = 0.2
ρ = 1 / (1 - σ)
δ = 0.5
β = 0.86
ϕ = 0.6
ξ = 0.5
μ = 0.5
ψ = 1 / ϕ
b = 1 / (ϕ^ϕ * (1 - ϕ)^(1 - ϕ))
ι = ϕ * ρ

numSectors = 2
numLin = 10
gridMin = 2
gridMax = 30

u(x,y) = (log(x) + log(y))
wInit(x,y) = (log(x) + log(y)) / (1 - β)
function gradwInit(g, x, y)
    g[1] = 1 / x / (1 - β)
    g[2] = 1 / y / (1 - β)
end
function hesswInit(h, x, y)
    h[1,1] = -1 / x^2 / (1 - β)
    h[1,2] = 0
    h[2,1] = 0
    h[2,2] = -1 / y^2 / (1 - β)
end
rng = MersenneTwister(1234)
# for timing
tOGrd = TimerOutput()
# wval = w.(grid)

# the following function generates target values (of the value function) for the ML process
##note that (unlike Sargent and Stachursky) the grid is written in terms of kapital
# this simplified the problem substantially as otherwise intermediate variables became dynamic
##also note that this accepts the smooth function
function TargGrd(w, grid, β ; return_policy=false)
    for n in 1:length(grid)
        @timeit tOGrd string("Pointwise-for-", length(grid), "-pts") begin
            k = grid[n]
            modTrial = Model(Ipopt.Optimizer)
#        set_silent(modTrial)
            c_strt = sum(grid[n]) / (3 * numSectors)
            y_strt = sum(grid[n]) / numSectors
            kp_strt = sum(grid[n]) / numSectors
            x_strt = sum(grid[n]) / (3 * numSectors^2)
            xC_strt = sum(grid[n]) / (3 * numSectors)
            m_strt = sum(grid[n]) / (3 * numSectors^2)
            mC_strt = sum(grid[n]) / (3 * numSectors)

            @variable(modTrial, 0.01 <= c[1:numSectors], start = c_strt)
            @variable(modTrial, 0.01 <= y[1:numSectors], start = y_strt)
            @variable(modTrial, 0.01 <= kp[1:numSectors], start = kp_strt)
            @variable(modTrial, 0.01 <= x[1:numSectors, 1:numSectors], start = x_strt)
            @variable(modTrial, 0.01 <= xC[1:numSectors], start = xC_strt)
            @variable(modTrial, 0.01 <= m[1:numSectors, 1:numSectors], start = m_strt)
            @variable(modTrial, 0.01 <= mC[1:numSectors], start = mC_strt)
            @variable(modTrial, u)
            @variable(modTrial, w, start = 2.0)
            for i in 1:numSectors
            # output
            # @NLconstraint(modTrial, y[i] == b * k[i]^ϕ * mC[i]^(1-ϕ))
                @NLconstraint(modTrial, y[i] == b * (ϕ * k[i]^ρ + (1 - ϕ) * mC[i]^ρ)^(1 / ρ))
            # future capital
                @constraint(modTrial, kp[i] == xC[i] + (1 - δ) * k[i])
            # supply equals demand (eq'm constraint)
                @constraint(modTrial, 0 == y[i] - c[i] - sum(m[i,:]) - sum(x[i,:]))
            end
            @NLconstraint(modTrial, xC[1] == x[1,1]^ξ * x[2,1]^(1 - ξ))
            @NLconstraint(modTrial, xC[2] == x[1,2]^ξ * x[2,2]^(1 - ξ))
            @NLconstraint(modTrial, mC[1] == m[1,1]^μ * m[2,1]^(1 - μ))
            @NLconstraint(modTrial, mC[2] == m[1,2]^μ * m[2,2]^(1 - μ))
#        @NLconstraint(modTrial, u == (γ*c[1]^ρ + (1-γ)*c[2]^ρ)^(1/ρ))
            @NLconstraint(modTrial, u == log(c[1]) + log(c[2]))
            @NLconstraint(modTrial, w == (log(kp[1]^ϕ) + log(kp[2]^(1 - ϕ))) / (1 - β))
#        @NLconstraint(modTrial, w == (γ*kp[1]^ι + (1-γ)*kp[2]^(ρ-ι))^(1/ρ)/(1-β))
#        register(modTrial, :wInit, 2, wInit, gradwInit)
#        @NLconstraint(modTrial, w == wInit(kp[1],kp[2]))
#        register(modTrial, :u, 2, u; autodiff = true)
            @NLobjective(modTrial, Max, u + β * w)
            @timeit tOGrd string("optimization-", length(grid), "-pts") begin
                JuMP.optimize!(modTrial)
            end
            push!(wTargGrd, JuMP.objective_value(modTrial))
            if return_policy
                push!(xTargGrd, value.(x))
                push!(sTargGrd, value.(c))
                push!(mTargGrd, value.(m))
                push!(yTargGrd, value.(y))
            end
        end # of timeit
#        global intK[n] = value.(f)
    end
    # global prevK = intK
    return wTargGrd
    end

function evalBellGrd(numLin)
    global yTargGrd = Vector{Float64}[]
    global sTargGrd = Vector{Float64}[]
    global mTargGrd = Matrix{Float64}[]
    global xTargGrd = Matrix{Float64}[]
    global wTargGrd = Float64[]
    global grid = Vector{Float64}[]
    for p in product(LinRange(gridMin, gridMax, numLin), LinRange(gridMin, gridMax, numLin))
        push!(grid, collect(p))
    end
    wTargGrd  = TargGrd(wInit, grid, β; return_policy=true)
    show(tOGrd)
    # reset containers and grid
    # tOCbd = TimerOutput()
    # yTargGrd = Vector{Float64}[]
    # sTargGrd = Vector{Float64}[]
    # mTargGrd = Matrix{Float64}[]
    # xTargGrd = Matrix{Float64}[]
    # wTargGrd = Float64[]
    # grid = Vector{Float64}[]
    # for p in product(LinRange(gridMin,gridMax,numLin), LinRange(gridMin, gridMax, numLin))
        # push!(grid, collect(p))
    # end
    println("\n ---------")
    return wTargGrd
    end
function chckEconGrd(numLin)
    global XTargGrd = Vector{Float64}[]
       for j = 1:numLin^numSectors
           push!(XTargGrd, sum(xTargGrd[j][:,i] for i in 1:numSectors))
       end
    global MTargGrd = Vector{Float64}[]
       for j = 1:numLin^numSectors
           push!(MTargGrd, sum(mTargGrd[j][:,i] for i in 1:numSectors))
       end
    global fTargGrd = sTargGrd + MTargGrd + XTargGrd

    global MPCf = Vector{Float64}[]
       for j = 1:numLin^numSectors
           push!(MPCf, sTargGrd[j] ./ fTargGrd[j])
       end
    global MPC = Vector{Float64}[]
       for j = 1:numLin^numSectors
           push!(MPC, sTargGrd[j] ./ yTargGrd[j])
       end
    global MPM = Vector{Float64}[]
       for j = 1:numLin^numSectors
           push!(MPM, MTargGrd[j] ./ yTargGrd[j])
       end
    global MPX = Vector{Float64}[]
       for j = 1:numLin^numSectors
           push!(MPX, XTargGrd[j] ./ yTargGrd[j])
       end
end
#=
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
gp = GP(gridM, wTarg+0.2*randn(numLin^numSectors), mz, kern)
#gp = GP(gridM, wTargGrd, MeanLin([2., 3.]), kern)
GaussianProcesses.optimize!(gp)
predict_f(gp, gridM)[1]
fstar(x) = predict_f(gp,hcat(x))[1]

plot!(contour(gp), heatmap(gp); fmt=:png)
#wTargp(x, y) = predict_f(gp,hcat([x, y]))[1] =#
