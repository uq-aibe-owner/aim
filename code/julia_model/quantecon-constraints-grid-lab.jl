
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
import Pkg; Pkg.add("TimerOutputs")
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
    Random,
    TimerOutputs;
    #DateTime;


δ = 0.1
β = 0.86
#ϕ = 0.6
ϕk = 0.3
ϕm = 0.3
ξ = 0.5
μ = 0.5
ψ = 1/ϕ
b = 1/(ϕ^ϕ * (1-ϕ)^(1-ϕ))
γ = 0.5


numSectors = 2
numLin = 10
gridMin = 2
gridMax = 30

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
yTargGrd = Vector{Float64}[]
sTargGrd = Vector{Float64}[]
lTargGrd = Vector{Float64}[]
mTargGrd = Matrix{Float64}[]
xTargGrd = Matrix{Float64}[]
wTargGrd = Float64[]
grid = Vector{Float64}[]
for p in product(LinRange(gridMin,gridMax,numLin), LinRange(gridMin, gridMax,numLin))
    push!(grid, collect(p))
end
rng = MersenneTwister(1234)
#for timing
tOGrd = TimerOutput()
#wval = w.(grid)
    
#the following function generates target values (of the value function) for the ML process
##note that (unlike Sargent and Stachursky) the grid is written in terms of kapital
#this simplified the problem substantially as otherwise intermediate variables became dynamic
##also note that this accepts the smooth function
function TargGrd(w, grid, β ; return_policy = false)
    for n in 1:length(grid)
        @timeit tOGrd string("Pointwise-for-", length(grid),"-pts") begin
        k = grid[n]
        modTrial = Model(Ipopt.Optimizer)
        set_silent(modTrial)
        @variable(modTrial, c[1:numSectors] >= 0.0001)
        @variable(modTrial, y[1:numSectors] >= 0.0001)
        @variable(modTrial, kp[1:numSectors]>= 0.0001)
        @variable(modTrial, x[1:numSectors, 1:numSectors]>=0.0001)
        @variable(modTrial, xComb[1:numSectors]>=0.0001)
        @variable(modTrial, m[1:numSectors, 1:numSectors]>=0.0001)
        @variable(modTrial, mComb[1:numSectors]>=0.0001)
        @variable(modTrial, lab[1:numSectors]>=0.0001)
        @variable(modTrial, u)
        @variable(modTrial, w)
        for i in 1:numSectors
            #output
            @NLconstraint(modTrial, y[i] == b * k[i]^ϕk * mComb[i]^ϕm * lab[i]^(1-ϕm-ϕk))
            #future capital
            @constraint(modTrial, kp[i] == xComb[i] + (1-δ) * k[i])
            #supply equals demand (eq'm constraint)
            @constraint(modTrial, 0 == y[i] - c[i] - sum(m[i,:]) - sum(x[i,:])) #done
        end
        @NLconstraint(modTrial, xComb[1] == x[1,1]^ξ * x[2,1]^(1-ξ))
        @NLconstraint(modTrial, xComb[2] == x[1,2]^ξ * x[2,2]^(1-ξ))
        @NLconstraint(modTrial, mComb[1] == m[1,1]^μ * m[2,1]^(1-μ))
        @NLconstraint(modTrial, mComb[2] == m[1,2]^μ * m[2,2]^(1-μ))
        @NLconstraint(modTrial, u == γ*log(c[1]) + (1-γ)*log(c[2]) - sum(lab[i] for i in 1:numSectors)^2)
        @NLconstraint(modTrial, w == (log(kp[1]) + log(kp[2]))/(1-β))
#        register(modTrial, :wInit, 2, wInit, gradwInit)
#        @NLconstraint(modTrial, w == wInit(kp[1],kp[2]))
#        register(modTrial, :u, 2, u; autodiff = true)
        @NLobjective(modTrial, Max, u + β * w)
        JuMP.optimize!(modTrial)
        push!(wTargGrd, JuMP.objective_value(modTrial))
        if return_policy
           push!(xTargGrd, value.(x))
           push!(sTargGrd, value.(c))
           push!(mTargGrd, value.(m))
           push!(yTargGrd, value.(y))
           push!(lTargGrd, value.(lab))
        end
                        end #of timeit
#        global intK[n] = value.(f)
    end
    #global prevK = intK
    return wTargGrd
end

function evalBellGrd()
    wTargGrd  = TargGrd(wInit, grid, β; return_policy= true)
    show(tOGrd)
    #reset containers and grid
    #tOCbd = TimerOutput()
    #yTargGrd = Vector{Float64}[]
    #sTargGrd = Vector{Float64}[]
    #mTargGrd = Matrix{Float64}[]
    #xTargGrd = Matrix{Float64}[]
    #wTargGrd = Float64[]
    #grid = Vector{Float64}[]
    #for p in product(LinRange(gridMin,gridMax,numLin), LinRange(gridMin, gridMax, numLin))
        #push!(grid, collect(p))
    #end
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
XTargGrd = Vector{Float64}[]
       for j = 1:numLin^numSectors
           push!(XTargGrd, sum(xTargGrd[j][:,i] for i in 1:numSectors))
       end
MTargGrd = Vector{Float64}[]
       for j = 1:numLin^numSectors
           push!(MTargGrd, sum(mTargGrd[j][:,i] for i in 1:numSectors))
       end
fTargGrd = sTargGrd + MTargGrd + XTargGrd

MPCf = Vector{Float64}[]
       for j = 1:numLin^numSectors
           push!(MPCf, sTargGrd[j]./fTargGrd[j])
       end
MPC = Vector{Float64}[]
       for j = 1:numLin^numSectors
           push!(MPC, sTargGrd[j]./yTargGrd[j])
       end
MPM = Vector{Float64}[]
       for j = 1:numLin^numSectors
           push!(MPM, MTargGrd[j]./yTargGrd[j])
       end
MPX = Vector{Float64}[]
       for j = 1:numLin^numSectors
           push!(MPX, XTargGrd[j]./yTargGrd[j])
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
gp = GP(gridM, wTarg+0.2*randn(numLin^numSectors), mz, kern)
#gp = GP(gridM, wTargGrd, MeanLin([2., 3.]), kern)
GaussianProcesses.optimize!(gp)
predict_f(gp, gridM)[1]
fstar(x) = predict_f(gp,hcat(x))[1]

plot!(contour(gp), heatmap(gp); fmt=:png)
#wTargp(x, y) = predict_f(gp,hcat([x, y]))[1]
=#
