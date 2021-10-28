
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
import Pkg; Pkg.add("BenchmarkTools")=#

using MathOptInterface, BenchmarkTools, LinearAlgebra, Statistics, QuantEcon, Interpolations, NLsolve, Optim, Random, IterTools, JuMP, Ipopt;


δ = 0.1
β = 0.96
ϕ = 0.6
ξ = 0.5
μ = 0.5
ψ = 1/ϕ
b = 1/(ϕ^ϕ * (1-ϕ)^(1-ϕ))


numSectors = 2
numPoints1D = 5
gridMax = 50
gridMin = 10

u(x,y) = (log(x) + log(y))
wInit(x,y) = (log(x) + log(y))/(1 - β)
grid = Vector{Float64}[]
yTarg = Vector{Float64}[]
sTarg = Vector{Float64}[]
mTarg = Matrix{Float64}[]
xTarg = Matrix{Float64}[]

for p in product(LinRange(gridMin,gridMax,numPoints1D),LinRange(gridMin,gridMax,numPoints1D))
    push!(grid, collect(p))
end

wTarg = Float64[]
#wval = w.(grid)
    
#the following function generates target values (of the value function) for the ML process
##note that (unlike Sargent and Stachursky) the grid is written in terms of kapital
#this simplified the problem substantially as otherwise intermediate variables became dynamic
##also note that this accepts the smooth function
function Targ(w, grid, β ; compute_policy = false)

    for n in 1:length(grid)
        k = grid[n]
        modTrial = Model(Ipopt.Optimizer);
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
            @constraint(modTrial, c[i] <= y[i])
            @NLconstraint(modTrial, y[i] == b * k[i]^ϕ * mComb[i]^(1-ϕ))
            @constraint(modTrial, kp[i] == xComb[i] + (1-δ) * k[i])
            @constraint(modTrial, 0 == y[i] - c[i] - sum(m[i,:]) - sum(x[i,:]))
        end
        @NLconstraint(modTrial, xComb[1] == x[1,1]^ξ * x[2,1]^(1-ξ))
        @NLconstraint(modTrial, xComb[2] == x[1,2]^ξ * x[2,2]^(1-ξ))
        @NLconstraint(modTrial, mComb[1] == m[1,1]^μ * m[2,1]^(1-μ))
        @NLconstraint(modTrial, mComb[2] == m[1,2]^μ * m[2,2]^(1-μ))
        @NLconstraint(modTrial, u == log(c[1]) + log(c[2]))
        @NLconstraint(modTrial, w == (log(kp[1]) + log(kp[2]))/(1-β))
#        register(modTrial, :w, 2, w; autodiff = true)
#        register(modTrial, :u, 2, u; autodiff = true)
        @NLobjective(modTrial, Max, u + β * w)
        optimize!(modTrial)
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
    return wTarg
end

wTargC  = Targ(wInit, grid, β; compute_policy=true)

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
IntPf = Vector{Float64}[]
       for j = 1:numPoints1D^numSectors
           push!(IntPf, MTarg[j]./fTarg[j])
       end
InvPf = Vector{Float64}[]
       for j = 1:numPoints1D^numSectors
           push!(InvPf, XTarg[j]./fTarg[j])
       end

#=
function solveOptGrowth(initial_w; tol = 1e-6, max_iter = 500)
    fixedpoint(w -> T(wVal, grid, β, f), initial_w).zero # gets returned
end
vStarApprox = solveOptGrowth(wVal)
=#

#=
u(c) = log(c[1]^0.5*c[2]^0.5);

numSectors = 2;


w(x,y) = x^2+y^2;

valGrid = zeros(numPoints1D,numPoints1D)
for i in 1:numPoints1D
    for j in 1:numPoints1D
        valGrid[i,j] = w(grid[j+(i-1)*numPoints1D][1],grid[j+(i-1)*numPoints1D][2])
    end
end

valGrid = zeros(numPoints1D,numPoints1D)
for i in 1:numPoints1D
    for j in 1:numPoints1D
        valGrid[i,j] = w(grid[j+(i-1)*numPoints1D][1],grid[j+(i-1)*numPoints1D][2])
    end
end

α = 0.4
β = 0.96
μ = 0
s = 0.1
ϵ_D = 1
ϵ_X = 0.1
Γ = [0.7 0.3; 0.3 0.7]

#initial capital flows
x= [1 2; 3 4]
#Investment is a function from R^2J to R^J
X = zeros(numSectors,1)
XInter = zeros(numSectors,numSectors)
for J in numSectors;
    for I in numSectors;
        XInter[I,J] = (Γ[I,J]^(1/ϵ_X)*x[I,J]^((ϵ_X-1)/ϵ_X))
    end
    X[J] = sum(XInter[:,J])^(ϵ_X/(ϵ_X-1))
end

K = zeros(numSectors,1)
Q = zeros(numSectors,1)
C = zeros(numSectors,1)

#Kapital is function from R^J to R^J
K = X;

Q = K.^α

C = Q - sum(x[:,J] for J in 1:numSectors)

#=
c1 = log(1 - α * β) / (1 - β)
c2 = (μ + α * log(α * β)) / (1 - α)
c3 = 1 / (1 - β)
c4 = 1 / (1 - α * β)
=#

# Utility is a function from R^J to R
u(C) = log(sum(C[j]^((ϵ_D-1)/ϵ_D) for j in 1:numSectors)^((ϵ_D)/(ϵ_D*0.999-1)));

# Deterministic part of production function a function from R^J to R^J
f(k) = k.^α

#= True optimal policy
c_star(y) = (1 - α * β) * y

# True value function
v_star(y) = c1 + c2 * (c3 - c4) + c4 * log(y)
=#

Random.seed!(42) # For reproducible results.

#=
grid = Array{Float64}[]
for n in 1:numSectors
    x = 1:numSectors
    push!(grid, x)
end
=#


#=
grid[j] = 1:numSectors
y = 1:numSectors
x' .* ones(5)
ones(3)' .* y
g         # Largest grid point
grid_size = 200      # Number of grid points
shock_size = [250,250]     # Number of shock draws in Monte Carlo integral

grid_y = range.(1e-5,  gridMax, length = grid_size)
shocks = exp.(μ .+ s * randn(shock_size))
w = T(v_star.(grid_y), grid_y, β, (log∘sum), k -> k^α, shocks)

#=
plt = plot(ylim = (-35,-24))
plot!(plt, grid_y, w, linewidth = 2, alpha = 0.6, label = "T(v_star)")
plot!(plt, v_star, grid_y, linewidth = 2, alpha=0.6, label = "v_star")
plot!(plt, legend = :bottomright)
=#
w = 5 * log.(grid_y)  # An initial condition -- fairly arbitrary
n = 35
#=
plot(xlim = (extrema(grid_y)), ylim = (-50, 10))
lb = "initial condition"
plt = plot(grid_y, w, color = :black, linewidth = 2, alpha = 0.8, label = lb)
for i in 1:n
    global w = T(w, grid_y, β, log, k -> k^α, shocks)
    plot!(grid_y, w, color = RGBA(i/n, 0, 1 - i/n, 0.8), linewidth = 2, alpha = 0.6,
          label = "")
end

lb = "true value function"
plot!(plt, v_star, grid_y, color = :black, linewidth = 2, alpha = 0.8, label = lb)
plot!(plt, legend = :bottomright)
=#

function solve_optgrowth(initial_w; tol = 1e-6, max_iter = 500)
    fixedpoint(w -> T(w, grid_y, β, u, f, shocks), initial_w).zero # gets returned
end

#=
initial_w = 5 * log.(grid_y)
v_star_approx = solve_optgrowth(initial_w)

plt = plot(ylim = (-35, -24))
plot!(plt, grid_y, v_star_approx, linewidth = 2, alpha = 0.6,
      label = "approximate value function")
plot!(plt, v_star, grid_y, linewidth = 2, alpha = 0.6, label = "true value function")
plot!(plt, legend = :bottomright)

Tw, σ = T(v_star_approx, grid_y, β, log, k -> k^α, shocks;
                         compute_policy = true)
cstar = (1 - α * β) * grid_y

plt = plot(grid_y, σ, lw=2, alpha=0.6, label = "approximate policy function")
plot!(plt, grid_y, cstar, lw = 2, alpha = 0.6, label = "true policy function")
plot!(plt, legend = :bottomright)
=#
=#
=#
