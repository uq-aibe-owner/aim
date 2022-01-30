using NLsolve, BenchmarkTools

# number of sectors
numSectors = 4

# test initial output statespace
y = 9.0 .+ rand(Float64, numSectors)

# linear production function parameter
ϕ = 0.5
# time discount factor
β = 0.96
# elasticity parameter
ρ = -2
# depreciation
δ = 0.2
# savings scaling parameters
σ = (1 / numSectors .* ones(numSectors, numSectors)).^(1-ρ)
#
η = 0.5

# test initial capital
k = y ./ ϕ

# utility function of consumption vector
u(c) = sum(log.(c))
# derivative of utility w.r.t consumption in one element
u′(c) = 1 ./ c

# production function of capital
f(k) = ϕ * k.^η
# derivative of production w.r.t capital
f′(k) = ϕ * η * k.^(η-1)
# inverse
fInv(y) = (1/ϕ * y).^(1/η)

# future capital from current parameters
kp(s,y) = (1 - δ) * fInv(y) + s

# this would be replaced by our interpolators in each sector in the iterations
# a g function of the current below form is equivalent to our "first guess" of an interpolator
g(y) = 1 / (numSectors) .* ones(2*numSectors-1) + rand(Float64, 2*numSectors-1)

# S_ij as function of S39
function S(S39)
    S = ones(numSectors, numSectors)

    # fills in the known values first (the diagonal and end collumn)
    for i in 1:numSectors
        for j in 1:numSectors
            if j == numSectors
                S[i,j] = S39[i]
            elseif i == j
                S[i,j] = S39[numSectors + i]
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

# consumption as function of S_ij ∀ i,j
c(S,y) = y .- sum(S, dims = 2)

# NLsolve approach
function f!(F, S39)
    for i in 1:2*numSectors-1
        if i <= numSectors
            # Nth (last) collumn of double blackboard S matrix, excluding diagonal corner
            F[i] = u′(c(S(S39), y))[numSectors] / u′(c(S(S39), y))[i] - Sb(S(S39))[i,numSectors]/Sb(S(S39))[numSectors,numSectors]
        else
            # Diagonals of S matrix
            j = i - numSectors
            F[i] = (f′(kp(s(S(S39)), y))[j] * β * u′(c(S(g(f(kp(s(S(S39)), y)))), f(kp(s(S(S39)), y))))[j] / u′(c(S(S39), y))[j]) - (S39[i] / (σ[j,j] * s(S(S39))[j])).^(1-ρ) # .^(1 / (1 - ρ)) put back inside exponent
        end
    end
end

# warm start for savings values
initial_x = 1/(numSectors*3) .* ones(2*numSectors-1).*100

# solve
nlsolve(f!, initial_x)

#@btime 