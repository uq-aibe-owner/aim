using JuMP, Ipopt, BenchmarkTools

# number of sectors
numSectors = 3

# test initial capital
k = 0.9*ones(numSectors)

# test initial output statespace
y = ones(numSectors)

# linear production function parameter
ϕ = 0.5
# time discount factor
β = 1
# elasticity parameter
ρ = -9
# depreciation
δ = 1
# savings scaling parameters
σ = 1 / numSectors .* ones(numSectors, numSectors)

# utility function of consumption vector
u(c) = sum(log.(c))
# derivative of utility w.r.t consumption in one element
u′(c) = 1 / c

# production function of capital
f(k) = ϕ * k
# derivative of production w.r.t capital
f′(k) = ϕ

# future capital from current parameters
kp(s,k) = (1 - δ) * k + s

# this would be replaced by our interpolators in each sector in the iterations
# a g function of the current below form is equivalent to our "first guess" of an interpolator
g(y) = 0.5 .* ones(2*numSectors-1)

# consumption as function of S_ij ∀ i,j
c(S,y) = y .- sum(S, dims = 2)

# S_ij as function of S39
function S(S39)
    S = ones(numSectors, numSectors)

    # fills in the known values first (the diagonal and end collumn)
    for i in 1:numSectors
        for j in 1:numSectors
            if i == j
                S[i,j] = S39[numSectors - 1 + i]
            elseif j == numSectors
                S[i,j] = S39[i]
            end
        end
    end
    # make partial Sb from the know values of S above
    Sb = S ./ σ

    # fills in the empty values in S using our partial Sb
    for i in 1:numSectors
        for j in 1:numSectors
            if i != j && j != numSectors
                S[i,j] = σ[i,j] * Sb[j,j] * Sb[i,numSectors] / Sb[j,numSectors]
            end
        end
    end

    # make full Sbb
    Sbb = ones(numSectors, numSectors)
    for i in 1:numSectors
        for j in 1:numSectors
            Sbb[i,j] = Sb[i,j] / Sb[j,j]
        end
    end

    # return savings matrix
    return S
end

Sb(S) = S ./ σ


# s_j from full S_ij
s(S) = (sum(σ[l,:] .* S[l,:].^ρ for l in 1:numSectors)).^(1/ρ)

j(i) = i - numSectors + 1

model = Model(Ipopt.Optimizer)
set_silent(model)
@variable(model, S39[1:2*numSectors-1] >= 0.00001)
@variable(model, F[1:2*numSectors-1])
register(model, :u′, 1, u′; autodiff = true)
for i in 1:(2*numSectors-1)
    if i < numSectors
        @NLconstraint(model, F[i] == u′(c(S(S39), y)[numSectors]) / u′(c(S(S39), y)[i]) - Sb(S(S39))[i,numSectors]/Sb(S(S39))[numSectors,numSectors])
    else
        # Diagonals of S matrix
        @NLconstraint(model, F[i] == ((f′(β) * β * u′(c(S(g(y)), kp(s(S(S39)), k))[j(i)])) / (u′(c(S(S39), y)[numSectors])))^(1 / (1 - ρ)) - S39[i] / (σ[j(i),j(i)] * s(S(S39))[j(i)])) # -ρ / (1 - ρ) put back inside exponent
    end
end
@NLobjective(model, Min, sum((F[i])^2 for i in 1:2*numSectors-1))
JuMP.optimize!(model)
println(value.(F))