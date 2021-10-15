
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
import Pkg; Pkg.add("Ipopt")=#

using MathOptInterface, LinearAlgebra, Statistics, QuantEcon, Interpolations, NLsolve, Optim, Random, IterTools, JuMP, Ipopt;

u(x) = sum(log(x[i]) for i in 1:numSectors)
w(x) = sum(log(x[i]) for i in 1:numSectors)

β = 0.96
α = 0.4
f(x,m) = x^α*m^(1-α)
γx = 0.5
γm = 0.5

numSectors = 2 
numPoints1D = 20 

grid = Vector{Vector{Float64}}(undef,(numPoints1D)^numSectors)

gridMax = 5
gridMin = 1
gridHood = 0

iter=1
for p in product(LinRange(gridMin,gridMax,numPoints1D),LinRange(gridMin,gridMax,numPoints1D))
    grid[iter] = collect(p)
    global iter += 1
end

wVal = w.(grid)
    
wVals = zeros(numPoints1D,numPoints1D);
for j in numPoints1D
    for i in numPoints1D
        wVals[i,j] = wVal[i+(i-1)*j]
    end
end


function T(wVal, grid, β, f ; compute_policy = false)
    wVals = zeros(numPoints1D,numPoints1D);
    for j in numPoints1D
        for i in numPoints1D
            wVals[i,j] = wVal[i+(i-1)*j]
        end
    end
    wFunc(x, y) = extrapolate(interpolate((LinRange(gridMin,gridMax,numPoints1D),LinRange(gridMin,gridMax,numPoints1D)), wVals, Gridded(Linear())), Interpolations.Flat())(x,y)
    global Tw = zeros(length(grid))
    global σ = similar(grid)
    global intK = similar(grid)
    for n in 1:length(grid)
        y = grid[n]
        prevK = y.^(1/α)
        modTrial = Model(Ipopt.Optimizer);
        @variable(modTrial,  c[1:numSectors] >= 0.0001)
        @variable(modTrial, k[1:numSectors])
        @variable(modTrial, x[1:numSectors, 1:numSectors]>=0.0001)
        @variable(modTrial, xSum[1:numSectors]>=0.0001)
        @variable(modTrial, m[1:numSectors, 1:numSectors]>=0.0001)
        @variable(modTrial, mSum[1:numSectors]>=0.0001)
        setvalue.(k, prevK)      
        for i in 1:numSectors
            @constraint(modTrial, gridMin <= c[i] <= y[i])
            @constraint(modTrial, k[i] == xSum[i] + (1-δk)*prevK[i])
        end
        @NLconstraint(modTrial, xSum[1] == x[1,1]^γx*x[2,1]^(1-γx))
        @NLconstraint(modTrial, xSum[2] == x[1,2]^γx*x[2,2]^(1-γx))
        @NLconstraint(modTrial, mSum[1] == m[1,1]^γm*m[2,1]^(1-γm))
        @NLconstraint(modTrial, mSum[2] == m[1,2]^γm*m[2,2]^(1-γm))
        @constraint(modTrial, x[1,1] == y[1] - c[1] -sum(m[1,:]) - x[1,2])
        @constraint(modTrial, x[2,2] == y[2] - c[2] -sum(m[2,:]) - x[2,1])
        @constraint(modTrial, x[1,2] == y[1] - c[1] -sum(m[1,:]) - x[1,1])
        @constraint(modTrial, x[2,1] == y[2] - c[2] -sum(m[2,:]) - x[2,2])

        register(modTrial, :wFunc, 2, wFunc; autodiff = true)
        register(modTrial, :f, 2, f; autodiff = true)
        @NLobjective(modTrial, Max, sum(log(c[i]) for i in 1:numSectors) + β*wFunc(f(k[1],mSum[1]),f(k[2],mSum[2])))
        optimize!(modTrial)
        Tw[n] = JuMP.objective_value(modTrial)
        if compute_policy
            σ[n] = value.(c)
        end
        global intK[n] = value.(k)
    end
    global prevK = intK
    if compute_policy
        return Tw, σ
    end
    return Tw
end

δk = 0.1
#wVal = T(wVal, grid, β, f; compute_policy = true)
function solveOptGrowth(initial_w; tol = 1e-6, max_iter = 500)
    fixedpoint(w -> T(wVal, grid, β, f), initial_w).zero # gets returned
end
vStarApprox = solveOptGrowth(wVal)
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
