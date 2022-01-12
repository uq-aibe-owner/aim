using LinearAlgebra, Statistics, BenchmarkTools, Interpolations, Parameters, Plots, QuantEcon, Roots, Optim, Random, IterTools, BasicInterpolators

function K!(Kg, g, grid, β, ∂u∂c, f, f′, shocks)
# This function requires the container of the output value as argument Kg
    # Construct radial basis functions
    g_func = RBFInterpolator(grid, vec(g), 1)
    # solve for updated consumption value
    for i in 1:length(grid[:,1])
        y=grid[i,:]
        function h(c) # implies c is similar object to grid, this is not compatible with our root finding method currently
            vals = ∂u∂c.(g_func.(sum(f.(y - c), dims=2)  * shocks)) .* f′(y - c) .* shocks
            return ∂u∂c(c) - β * mean(vals)
        end
        Kg[i] = find_zero(h, (1e-10, y .- 1e-10)) # would need to make this N (or at least 2-dimensional for a test)
    end
    return Kg
end
    
# The following function does NOT require the container of the output value as argument
K(g, grid, β, ∂u∂c, f, f′, shocks) = K!(similar(g), g, grid, β, ∂u∂c, f, f′, shocks)

# returns a cartesian product matrix that is compatible with the RBF interpolation
function cartProd(gridMin,gridMax,gridSize, numSec)
    # creates a vector of all the ranges we need
    ranges = fill(LinRange(gridMin,gridMax,gridSize),numSec)
    # initialise empty array
    grid = Array{Float64}(undef,gridSize^numSec,numSec)
    for (index, p) in enumerate(product(ranges...))
        # coordinates in each dimension are across rows
        grid[index, :] = collect(p)
    end   
    return grid
end

function verify_true_policy(m, shocks, c_star)
    # compute (Kc_star)
    @unpack grid, β, ∂u∂c, f, f′ = m
    c_star_new = K(c_star, grid, β, ∂u∂c, f, f′, shocks)

    #compare
    return (c_star-c_star_new)./c_star
end

function check_convergence(m, shocks, c_star, g_init; n_iter = 15)
    @unpack grid, β, ∂u∂c, f, f′ = m
    g = g_init;
    for i in 1:n_iter
        new_g = K(g, grid, β, ∂u∂c, f, f′, shocks)
        g = new_g
    end
    #compare
    return (c_star-g)./c_star
end

Model=@with_kw (α = 0.65,                            # Productivity parameter
                β = 0.95,                            # Discount factor
                γ = 1.0,                             # Risk aversion
                μ = 0.0,                             # First parameter in lognorm(μ, σ)
                s = 0.1,                             # Second parameter in lognorm(μ, σ)
                numSec = 2,
                gridMin = 1e-6,                     # Smallest grid point
                gridMax = 4.0,                      # Largest grid point
                gridSize = 20,                     # Number of grid points
                grid = cartProd(gridMin, gridMax, gridSize, numSec), # Grid
                u = (c, γ = γ) -> isoelastic(c, γ),  # utility function
                ∂u∂c = c -> c^(-γ),                  # u′
                f = k -> k^α,                        # production function
                f′ = k -> α * k^(α - 1),             # f′
                )


m = Model();

Random.seed!(42) # for reproducible results.
shock_size = 250 # number of shock draws in Monte Carlo integral
shocks = collect(exp.(m.μ .+ m.s * randn(shock_size))); # generate shocks

c_star = sum((1 - m.α * m.β) .* m.grid, dims=2) # true policy (c_star)
verify_true_policy(m, shocks, c_star)

isoelastic(c, γ) = isone(γ) ? log(c) : (c^(1 - γ) - 1) / (1 - γ)

check_convergence(m, shocks, c_star, m.grid, n_iter = 15)