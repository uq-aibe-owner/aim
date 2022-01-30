using JuMP, Ipopt, BenchmarkTools

# number of sectors
numSectors = 4

# test initial output statespace
y = 9.0 .+ rand(Float64, numSectors)
#
Ε = 0.1
#
ρ = (Ε-1) / Ε
# savings scaling parameters
σ = (1 / numSectors .* ones(numSectors, numSectors)).^(1-ρ)
#
γ = ones(numSectors) ./ (2*numSectors)

model = Model(Ipopt.Optimizer)
set_silent(model)
@variable(model, Sb[1:numSectors] >= 0.00001)
@variable(model, F[1:numSectors])
@variable(model, Sbb[1:numSectors, 1:numSectors])
for j in 1:numSectors
    @NLconstraint(model, F[j] == (γ[numSectors] * y[j] - Sbb[j,numSectors]^Ε * γ[j] * y[numSectors]) - sum((σ[j,k] * Sbb[j,k] - σ[numSectors,k]*Sbb[numSectors, k])*Sb[k] for k in 1:numSectors))
    for l in 1:numSectors
        @NLconstraint(model, Sbb[l,j] == (γ[j] / γ[l] * y[l] / y[j])^(1/Ε))
    end
end
@NLobjective(model, Min, sum((F[i])^2 for i in 1:numSectors))
JuMP.optimize!(model)
println(value.(Sb))
