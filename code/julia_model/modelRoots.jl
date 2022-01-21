using NLsolve, BenchmarkTools

# linear production function parameter
ϕ = 0.5
# time discount factor
β = 0.9
# elasticity parameter
ρ = -10
# depreciation
δ = 1
# savings scaling parameters
σ = ones(numSectors, numSectors)

# utility function of consumption
u(c) = log(c)
# derivative of utility w.r.t consumption
u′(c) = 1/c

# production function of capital
f(k) = ϕ*k
# derivative of production w.r.t capital
f′(k) = ϕ

# future capital from current parameters
kp(y,k,c) = (1-δ)*k+y-c

# this would be replaced by our interpolators in each sector in the iterations
# a g function of the current below form is equivalent to our "first guess" of an interpolator
g(y) = 1

# consumption as function of S_ij ∀ i,j


# full S_ij from 39 element S_ij / bbS_ij


# NLsolve approach
function f!(F, c)
    for i in 1:2*numSectors-1
        if i <= numSectors
            # Nth collumn of double blackboard S matrix
            F[i] = Sb[i,j]/Sb[j,j] - u′(c[j])/u′(c[i])
        else
            # Diagonals of blackboard S matrix
            F[i] = (f′(β)*β*u′(g(f(kp(y,k[i],c[i]))))/(u′(c[i])))^(-rho/(1-rho))-sum(σ[l,j]*(u′[c[j]]) for l in 1:numSectors)
        end
    end
end

# warm start for consumption values
initial_x = 1.5*ones(numSectors)

# solve
nlsolve(f!, initial_x)
