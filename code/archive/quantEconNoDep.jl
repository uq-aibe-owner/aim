using MathOptInterface, LinearAlgebra, Statistics, Plots, QuantEcon, Interpolations, NLsolve, Optim, Random, IterTools, JuMP, Ipopt, MathOptInterface;

u(x) = sum(log(x[i]) for i in 1:numSectors)
w(x) = sum(5*log(x[i]) for i in 1:numSectors)

β = 0.96
α = 0.4
f(x) = x^α

numSectors = 2

numPoints1D = 2

grid = Vector{Vector{Float64}}(undef,(numPoints1D)^numSectors)

gridMax = 3
gridMin = 1

iter=1
if numSectors == 1
	grid = LinRange(gridMin, gridMax, numPoints1D)
elseif numSectors == 2
for p in product(LinRange(gridMin,gridMax,numPoints1D),LinRange(gridMin,gridMax,numPoints1D))
    grid[iter] = collect(p)
    global iter += 1
end
else
	print("Currently we only allow one or two sectors")
end
wVal = w.(grid)
    
function interp(x,y)
    return interpolate((LinRange(gridMin,gridMaxnumPoints1D),LinRange(gridMin,gridMax,numPoints1D)), wVals, Gridded(Linear()))(x,y)
end

function T(wVal, grid, β, f ; compute_policy = false)
	if numSectors == 1
		wVals = zeros(numPoints1D)
		wVals = wVal
		wFunc(y) = interpolate(grid)
	elseif numSectors == 2
		wVals = zeros(numPoints1D,numPoints1D);
		for j in numPoints1D
		    for i in numPoints1D
                        wVals[i,j] = wVal[i+(i-1)*j]
                    end
                end
		wFunc(x,y) = log(x) + log(y)
		#wFunc(x, y) = interpolate((LinRange(gridMin,gridMax,numPoints1D),LinRange(gridMin,gridMax,numPoints1D)),			  wVals, Gridded(Linear()))(x,y)
		print("wFunc created")
		print(wFunc(1.5,1.5))
	else
		print("Can only handle upto 2 sectors")
	end
    print(wFunc(1.5, 1.5))
    global Tw = zeros(length(grid))
    global σ = similar(grid)
    for n in 1:length(grid)
        y = grid[n]
        modTrial = Model(Ipopt.Optimizer);
        @variable(modTrial,  c[1:numSectors] >= 0.0001)
        @variable(modTrial, k[1:numSectors] >= 0.0001, start=2)
        for i in 1:numSectors
            @constraint(modTrial, gridMin <= c[i] <= y[i])
	    @constraint(modTrial, k[i] == y[i] - c[i])
        end
        register(modTrial, :wFunc, numSectors, wFunc; autodiff = true)
        register(modTrial, :f, 1, f; autodiff = true)
	if numSectors == 1
		@NLobjective(modTrial, Max, sum(log(c[i]) for i in 1:numSectors) + β*wFunc(f(k[1])))
	elseif numSectors == 2 
        @NLobjective(modTrial, Max, sum(log(c[i]) for i in 1:numSectors) + β*wFunc(f(k[1]),f(k[2])))
	else
	print("check the number of sectors")
	end
        optimize!(modTrial)
        Tw[n] = JuMP.objective_value(modTrial)
        if compute_policy
            σ[n] = value.(c)
        end
    end
    if compute_policy
        return Tw, σ
    end
    
    return Tw
    
end


wVal = T(wVal, grid, β, f; compute_policy = false)
#=
u(c) = log(c[1]^0.5*c[2]^0.5);

numSectors == 2;



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
