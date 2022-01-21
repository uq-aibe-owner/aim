using LinearAlgebra, Statistics
using BenchmarkTools, Interpolations, Parameters, Plots, QuantEcon, Random, Roots

function coleman_egm(g, k_grid, β, u′, u′_inv, f, f′, shocks)

    # Allocate memory for value of consumption on endogenous grid points
    c = similar(k_grid)

    # Solve for updated consumption value
    for (i, k) in enumerate(k_grid)
        vals = u′.(g.(f(k) * shocks)) .* f′(k) .* shocks
        c[i] = u′_inv(β * mean(vals))
    end

    # Determine endogenous grid
    y = k_grid + c  # y_i = k_i + c_i

    # Update policy function and return
    Kg = LinearInterpolation(y,c, extrapolation_bc=Line())
    Kg_f(x) = Kg(x)
    return Kg_f
end

Model = @with_kw (α = 0.65, # productivity parameter
                  β = 0.95, # discount factor
                  γ = 1.0,  # risk aversion
                  μ = 0.0,  # lognorm(μ, σ)
                  s = 0.1,  # lognorm(μ, σ)
                  grid_min = 1e-6, # smallest grid point
                  grid_max = 4.0,  # largest grid point
                  grid_size = 200, # grid size
                  u = γ == 1 ? log : c->(c^(1-γ)-1)/(1-γ), # utility function
                  u′ = c-> c^(-γ), # u'
                  f = k-> k*α, # production function
                  f′ = k -> α, # f'
                  grid = range(grid_min, grid_max, length = grid_size)) # grid


mlog = Model(); # Log Linear model

Random.seed!(42); # For reproducible behavior.

shock_size = 250     # Number of shock draws in Monte Carlo integral
shocks = exp.(mlog.μ .+ mlog.s * randn(shock_size));

c_star(y) = (1 - mlog.α * mlog.β) * y

# some useful constants
ab = mlog.α * mlog.β
c1 = log(1 - ab) / (1 - mlog.β)
c2 = (mlog.μ + mlog.α * log(ab)) / (1 - mlog.α)
c3 = 1 / (1 - mlog.β)
c4 = 1 / (1 - ab)

v_star(y) = c1 + c2 * (c3 - c4) + c4 * log(y)

n = 15
function check_convergence(m, shocks, c_star, g_init, n_iter)
    k_grid = m.grid
    g = g_init
    plt = plot()
    plot!(plt, m.grid, g.(m.grid),
          color = RGBA(0,0,0,1), lw = 2, alpha = 0.6, label = "initial condition c(y) = y")
    for i in 1:n_iter
        new_g = coleman_egm(g, k_grid, m.β, m.u′, m.u′, m.f, m.f′, shocks)
        g = new_g
        plot!(plt, k_grid, new_g.(k_grid), alpha = 0.6, color = RGBA(0,0,(i / n_iter), 1),
              lw = 2, label = "")
    end

    plot!(plt, k_grid, c_star.(k_grid),
          color = :black, lw = 2, alpha = 0.8, label = "true policy function c*")
    plot!(plt, legend = :topleft)
end

check_convergence(mlog, shocks, c_star, identity, n)