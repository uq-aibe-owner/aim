using MathOptInterface, LinearAlgebra, Statistics, Plots, QuantEcon, Interpolations, NLsolve, Optim, Random, IterTools, JuMP, Ipopt, MathOptInterface;

u(x) = sum(log(x[i]) for i in 1:numSectors)
w(x) = sum(5*log(x[i]) for i in 1:numSectors)

β = 0.96
α = 0.4
f(x) = x^α

numSectors = 2

numPoints1D = 2

grid = Vector{Vector{Float64}}(undef,(numPoints1D)^numSectors)

gridMax = 2
gridMin = 1

iter=1
for p in product(LinRange(gridMin,gridMax,numPoints1D),LinRange(gridMin,gridMax,numPoints1D))
    grid[iter] = collect(p)
    global iter += 1
end

wVal = w.(grid)
    

function T(wVal, grid, β, f ; compute_policy = false)

    wVals = zeros(numPoints1D,numPoints1D);
    for j in numPoints1D
        for i in numPoints1D
            wVals[i,j] = wVal[i+(i-1)*j]
        end
    end
    function wFunc(x,y)
        return interpolate((LinRange(gridMin,gridMax,numPoints1D),LinRange(gridMin,gridMax,numPoints1D)), wVals, Gridded(Linear()))(x,y)
    end
    Tw = zeros(length(grid))
    σ = similar(grid)
    for n in 1:length(grid)
        y = grid[n]
    end

    objectives = ( c -> sum(log(c[i]) for i in 1:numSectors) + β*wFunc(f(k[1]),f(k[2])))

    results = maximize.(objectives, 1e-10, grid) # solver result for each grid point
    Tw = Optim.maximum.(results)
    if compute_policy
        σ = Optim.maximizer.(results)
        return Tw, σ
    end

    return Tw

        #=
        modTrial = Model(Ipopt.Optimizer);
        @variable(modTrial,  c[1:numSectors])
        @variable(modTrial, k[1:numSectors], start=1.000001)
        for i in 1:numSectors
            @constraint(modTrial, 0.99*gridMin <= c[i] <= 0.999*y[i])
	    @constraint(modTrial, k[i] == y[i] - c[i])
        end
	register(modTrial, :wFunc, 2, wFunc; autodiff = true)
	register(modTrial, :f, 1, f; autodiff = true)
	@NLobjective(modTrial, Max, sum(log(c[i]) for i in 1:numSectors) + β*wFunc(f(k[1]),f(k[2])))
        optimize!(modTrial)
        Tw[n] = JuMP.objective_value(modTrial)
        if compute_policy
            σ[n] = value.(c)
            return Tw, σ
        end
    end
    
    return Tw
    =#
end 


wVal = T(wVal, grid, β, f)
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