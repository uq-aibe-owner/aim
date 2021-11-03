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
    Distances,
    Random;

numSectors = 2
numPoints1D = 25
gridMax = 50
gridMin = 10
grid = Vector{Float64}[]
for p in product(LinRange(gridMin,gridMax,numPoints1D),LinRange(gridMin,gridMax,numPoints1D))
    push!(grid, collect(p))
end
Kxx = Float64[]
kappa(a, b) = exp(-.5 * norm(a - b)^2)
xstar = [37., 37.]
for n in 1:length(grid)
    for m in 1:length(grid)
        x = grid[n]
        y = grid[m]
        push!(Kxx, kappa(x, y))
    end
end
Kxx = reshape(Kxx,numPoints1D^2,numPoints1D^2)
invKxx = inv(Kxx*Kxx)
#the following needs wTarg to have been generated somewhere (at this stage I have via quantecon-constraints.jl in mind)
f = wTarg

function mu(xstar)
Kxs = Float64[]
for n in 1:length(grid)
    x = grid[n]
    push!(Kxs, kappa(x,xstar))
end
return Kxs'*invKxx*f
end
