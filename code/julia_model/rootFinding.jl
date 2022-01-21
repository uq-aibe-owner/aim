using JuMP, Ipopt, NLsolve, BenchmarkTools

# Ipopt approach
numSectors = 25
u(c) = log(c)
modTrial = Model(Ipopt.Optimizer)
set_silent(modTrial)
@variable(modTrial, c[1:numSectors] >= 0.01)
@NLobjective(modTrial, Min, sum((u(c[i]))^2 for i in 1:numSectors))
@btime JuMP.optimize!(modTrial)
println(value.(c))

# NLsolve approach
function f!(F, x)
    for i in 1:numSectors
        F[i] = u(x[i]) 
    end
end
initial_x = 1.5*ones(numSectors)
@btime nlsolve(f!, initial_x)

#NLsolve seems faster
