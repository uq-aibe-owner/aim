using
    MathOptInterface,
    BenchmarkTools,
    Random,
    IterTools,
    GaussianProcesses,
    Plots
    Random;

func(x) = (x[1]-2)^2*x[1]+x[2];

numSamps = 10
numPlot = 20
sampGrid = Vector{Float64}[]
plotGrid = Vector{Float64}[]

for p in product(LinRange(0,3,numSamps), LinRange(0,3,numSamps))
    push!(sampGrid, collect(p))
end

for p in product(LinRange(0,3,numPlot), LinRange(0,3,numPlot))
    push!(plotGrid, collect(p))
end

trueFunc = func.(plotGrid)

mz = MeanZero()
mc = MeanConst(1.)
#ml = MeanLin()
mp = MeanPoly([2. 3.; 1. 2.])
kern = SE(0.0, 0.0)
sampGridT = reduce(hcat, sampGrid)
sampVals = func.(sampGrid)+0.1*randn(numSamps^2)
gp = GP(sampGridT, sampVals, mz, kern)
GaussianProcesses.optimize!(gp)

attemptGrid = zeros(numPlot,numPlot)
for i in 1:length(plotGrid)
    attemptGrid[Int(trunc((i-1)/numPlot))+1, i-Int(trunc((i-1)/numPlot))*numPlot] = predict_f(gp, hcat(plotGrid[i]))[1][1]
end
trueGrid = zeros(numPlot,numPlot)
for i in 1:length(plotGrid)
    trueGrid[Int(trunc((i-1)/numPlot))+1, i-Int(trunc((i-1)/numPlot))*numPlot] = func(plotGrid[i])
end

plot(trueGrid, st=:surface, label = "True Function")
plot!(attemptGrid, st=:surface, label = "GP Attempt")