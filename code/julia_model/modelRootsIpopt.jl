using NLsolve, BenchmarkTools

# number of sectors
numSectors = 3

# test initial capital
k = ones(numSectors)

# test initial output statespace
y = ones(numSectors)

# linear production function parameter
ϕ = 0.5
# time discount factor
β = 0.9
# elasticity parameter
ρ = 10
# depreciation
δ = 1
# savings scaling parameters
σ = ones(numSectors, numSectors)

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
g(y) = ones(2*numSectors-1)

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

modTrial = Model(Ipopt.Optimizer)
set_silent(modTrial)
@variable(modTrial, c[1:numSectors] >= 0.01)
@NLobjective(modTrial, Min, sum((u(c[i]))^2 for i in 1:numSectors))
@btime JuMP.optimize!(modTrial)
println(value.(c))

# NLsolve approach
function f!(F, S39)
    for i in 1:2*numSectors-1
        if i < numSectors
            # Nth (last) collumn of double blackboard S matrix, excluding diagonal corner
            F[i] = u′(c(S(S39), y)[numSectors]) / u′(c(S(S39), y)[i]) - Sb(S(S39))[i,numSectors]/Sb(S(S39))[numSectors,numSectors]
        else
            # Diagonals of S matrix
            j = i - numSectors + 1
            F[i] = (f′(β) * β * u′(c(S(g(y)), kp(s(S(S39)), k))[j])) / (u′(c(S(S39), y)[numSectors]))^(ρ) - S39[i] / (σ[j,j] * s(S(S39))[j]) # -ρ / (1 - ρ) put back inside exponent
        end
    end
end

# warm start for savings values
initial_x = 0.5*ones(2*numSectors-1)

# solve
nlsolve(f!, initial_x)
