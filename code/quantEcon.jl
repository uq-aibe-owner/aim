using LinearAlgebra, Statistics
using Plots, QuantEcon, Interpolations, NLsolve, Optim, Random
#include("reducedIO.jl")

function T(w, grid, β, u, f, shocks; compute_policy = false)
    #generalise to n dimensions
    applied = w.(grid)
    w_func = LinearInterpolation(grid, w)
    # objective for each grid point
    objectives = (c -> u(c) + β * mean(w_func.(f(y - c) .* shocks)) for y in grid)
    results = maximize.(objectives, 1e-10, grid) # solver result for each grid point

    Tw = Optim.maximum.(results)
    if compute_policy
        σ = Optim.maximizer.(results)
        return Tw, σ 
        end

    return Tw
end

numSectors = 2;

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
numPoints = 10
grid = AbstractVector{Tuple{Float64, Float64}}(undef,(numPoints*numSectors)^numSectors)

grid_max = 4
grid_min = 0.01
grid_space = (grid_max-grid_min)/numPoints
i=1


for p in product(grid_min:grid_space:grid_max,grid_min:grid_space:grid_max)
    grid[i] = p
    i += 1
end

#=
grid[j] = 1:numSectors
y = 1:numSectors
x' .* ones(5)
ones(3)' .* y
g         # Largest grid point
grid_size = 200      # Number of grid points
shock_size = [250,250]     # Number of shock draws in Monte Carlo integral

grid_y = range.(1e-5,  grid_max, length = grid_size)
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