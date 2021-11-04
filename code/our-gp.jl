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
    Plots,
    Random;

numSectors = 2
numPoints1D = 35
gridMax = 50
gridMin = 10
grid = Vector{Float64}[]
for p in product(LinRange(gridMin,gridMax,numPoints1D),LinRange(gridMin,gridMax,numPoints1D))
    push!(grid, collect(p))
end
A = 1
l = 0.1
σ = 0.8
Id = 1.0*Matrix(I, numPoints1D^2, numPoints1D^2)
kappa(a, b) = A * exp(-.5 * norm(a - b)^2/l)
Kxx = Float64[]
for n in 1:length(grid)
    for m in 1:length(grid)
        x = grid[n]
        y = grid[m]
        push!(Kxx, kappa(x, y))
    end
end
Kxx = reshape(Kxx,numPoints1D^2,numPoints1D^2)
invKxx = inv(Kxx*Kxx + σ^2 * Id)
#the following needs wTarg to have been generated somewhere (at this stage I have via quantecon-constraints.jl in mind)
f = wTarg

xstar = grid[6]
Kxs = Float64[]
for n in 1:length(grid)
    x = grid[n]
    push!(Kxs, kappa(x, xstar))
end

mustar = Kxs'*inv(Kxx*Kxx)*f

function mu(y)
Kxs = Float64[]
for n in 1:length(grid)
    x = grid[n]
    push!(Kxs, kappa(x, y))
end
return Kxs'*invKxx*wTarg
end
numPlot = 30
plotGrid = Vector{Float64}[]

for p in product(LinRange(gridMin, gridMax,numPlot), LinRange(gridMin, gridMax,numPlot))
    push!(plotGrid, collect(p))
end
attemptGrid = zeros(numPlot,numPlot)
for i in 1:length(plotGrid)
    attemptGrid[Int(trunc((i-1)/numPlot))+1, i-Int(trunc((i-1)/numPlot))*numPlot] = mu(plotGrid[i])
end
trueGrid = zeros(numPlot,numPlot)
for i in 1:length(plotGrid)
    trueGrid[Int(trunc((i-1)/numPlot))+1, i-Int(trunc((i-1)/numPlot))*numPlot] = fstar(plotGrid[i])[1]
end

plot(trueGrid, st=:surface, label = "True Function")
plot!(attemptGrid, st=:surface, label = "GP Attempt")
