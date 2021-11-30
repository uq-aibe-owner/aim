tOCbd = TimerOutput()
for i in 10:10:500
    global numLin = i
    global grid = Vector{Float64}[]
    for p in product(LinRange(gridMin, gridMax, numLin),LinRange(gridMin, gridMax, numLin))
        push!(grid, collect(p))
    end
    evalBellCbd()
end
