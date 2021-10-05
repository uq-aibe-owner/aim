using LinearAlgebra, Statistics
using Plots, QuantEcon, Interpolations, NLsolve, Optim, Random


function T(w, grid, β, u, f; compute_policy = false)
    w_func = w
    # objective for each grid point
    objectives(c) = u(c) + β * mean(w_func.(f(c)))
    global bobjectives = objectives
    #println("here's the objectives")
    #println(objectives)
    results = maximize(objectives, 1e-10, grid) # solver result for each grid point
    global besults = results
    #println(results)

    Tw = Optim.maximum.(results)
    if compute_policy
        σ = Optim.maximizer.(results)
        return Tw, σ
    end

    return Tw
end
α = 0.4
β = 0.96
μ = 0
s = 0.1

c1 = log(1 - α * β) / (1 - β)
c2 = (μ + α * log(α * β)) / (1 - α)
c3 = 1 / (1 - β)
c4 = 1 / (1 - α * β)

# Utility
u(c) = log(c)

∂u∂c(c) = 1 / c

# Deterministic part of production function
f(k) = k^α

f′(k) = α * k^(α - 1)

# True optimal policy
c_star(y) = (1 - α * β) * y

# True value function
v_star(y) = c1 + c2 * (c3 - c4) + c4 * log(y)
v(y) = c1 + c2 * (c3 - c4) + c4 * y
using Random
Random.seed!(42) # For reproducible results.

grid_max = 4         # Largest grid point
grid_size = 200      # Number of grid points
shock_size = 250     # Number of shock draws in Monte Carlo integral

grid = 4#range(1e-5,  grid_max, length = grid_size)
shocks = 0 #exp.(μ .+ s * randn(shock_size))
w = T(v(grid), grid, β, log, k -> k^α) 
#=plt = plot(ylim = (-35,-24))
plot!(plt, grid_y, w, linewidth = 2, alpha = 0.6, label = "T(v)")
plot!(plt, v_star, grid_y, linewidth = 2, alpha=0.6, label = "v_star")
plot!(plt, legend = :bottomright)=#