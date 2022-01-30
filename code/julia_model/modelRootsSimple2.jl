using NLsolve, BenchmarkTools, Random, LinearAlgebra

rng = MersenneTwister(1235)

# number of sectors
n = 4

# test initial output statespace
y = 7 .+ 3*rand(rng, n)
y = 2 .* y
#=
lastVal = y[n]
y[n] = y[1]
y[1] = lastVal
=#
Ε = 0.2
#
ρ = -2
#
ep2 = (1/(1-ρ))
#
μ = 0.2
# savings scaling parameters
σ = ones(n, n) 
randVecs = rand(rng, n, n + 1)
randVecs[:, 1:n] = randVecs[:, 1:n] + I(n) .* μ
for i in 1:n
    #randVecs[:,i] = rand(rng, n)
    #randVecs[:,i] = randVecs[:,i] ./ sum(randVecs[:,i].^(1-ρ))
    for j in 1:n
        σ[j,i] = randVecs[j,i]
    end
end

for i in 1:n
    σ[:,i] = σ[:,i] ./ sum(σ[:,i])
end

#
#randVecs[:, n + 1] = randVecs[:, n + 1] ./ sum(randVecs[:, n + 1])
γ = randVecs[:, n + 1]
γ = γ ./ sum(γ)


y = γ.^(Ε).*100 + 100*rand(rng, n)


Sbb = ones(n, n)
for j in 1:n
    for k in 1:n
        Sbb[k,j] = γ[j] / γ[k] * (y[k] / y[j])^(1/Ε)
    end
end
#=
for j in 1:n 
    for l in 1:n
        if l < j 
=#



t = 10

A = ones(n-1, n-1)
b = ones(n-1)

for j in 1:n-1 
    b[j] = (y[j]* σ[n,n].^ep2 * Sbb[n,n] - y[n] * σ[j,n].^ep2 * Sbb[j,n]) * t # we dont need any more?
    for k in 1:n-1
        A[j,k] = y[j] * σ[n,k].^ep2 * Sbb[n,k] - y[n] * σ[j,k].^ep2 * Sbb[j, k]
    end
end

println(A)

invA = inv(A)

x = invA*b 


SbDiags = append!(x, t)

Sb = ones(n,n)
for i in 1:n
    for j in 1:n
        Sb[i,j] = Sbb[i,j] * SbDiags[j]
    end
end

S = Sb .* σ.^ep2


#=

# NLsolve approach
function f!(F, Sb)
    for j in 1:n
        F[j] = (γ[n] * y[j] - Sbb[j,n]^Ε * γ[j] * y[n]) - sum((σ[j,k] * Sbb[j,k] - σ[n,k]*Sbb[n, k])*abs(Sb[k]) for k in 1:n)
    end
end

# warm start for savings values
initial_x = ones(n) ./ n

# solve
nlsolve(f!, initial_x)

#@btime 
=#