using Pkg; 
Pkg.add("Plots") 
Pkg.add("LinearAlgebra")
Pkg.add("Statistics")
Pkg.add("BenchmarkTools")
Pkg.add("Interpolations")
Pkg.add("Parameters")
Pkg.add("QuantEcon")
Pkg.add("Roots")
Pkg.add("Optim")
Pkg.add("Random")
using LinearAlgebra, Statistics
using BenchmarkTools, Interpolations, Parameters, Plots, QuantEcon, Roots
using Optim, Random
gr(fmt = :png);

function K!(Kg, g, grid, β, ∂u∂c, f, f′, shocks)
# This function requires the container of the output value as argument Kg

    # Construct linear interpolation object
    g_func = LinearInterpolation(grid, g, extrapolation_bc = Line())

    # solve for updated consumption value
    for (i, y) in enumerate(grid)
        function h(c)
            vals = ∂u∂c.(g_func.(f(y - c) * shocks)) .* f′(y - c) .* shocks
            return ∂u∂c(c) - β * mean(vals)
        end
        Kg[i] = find_zero(h, (1e-10, y - 1e-10))
    end
    return Kg
end

# The following function does NOT require the container of the output value as argument
K(g, grid, β, ∂u∂c, f, f′, shocks) =
    K!(similar(g), g, grid, β, ∂u∂c, f, f′, shocks)

function T(w, grid, β, u, f, shocks, Tw = similar(w);
		compute_policy = false)

	# apply linear interpolation to w
	w_func = LinearInterpolation(grid, w, extrapolation_bc = Line())

	if compute_policy
		σ = similar(w)
	end

	# set Tw[i]  = max_c { u(c) + β E w(f(y - c) z) }
	for (i, y) in enumerate(grid)
		objective(c) = u(c) + β * mean(w_func.(f(y - c) .* shocks))
		res = maximize(objective, 1e-10, y)

		if compute_policy
			σ[i] = Optim.maximizer(res)
		end
		Tw[i] = Optim.maximum(res)
	end

	if compute_policy
		return Tw, σ
	else
		return Tw
	end
end


isoelastic(c, γ) = isone(γ) ? log(c) : (c^(1 - γ) - 1) / (1 - γ)
Model  = @with_kw (α = 0.65,				#Productivity parameter
		   β = 0.95,				#Discount factor
		   γ = 1.0,				#Risk aversion
		   μ = 0.0,				#First parameter in lognorm(μ, σ)
		   s = 0.1,				#second parameter in lognorm
		   grid = range(1e-6, 4, length = 200),	#Grid
		   grid_min = 1e-6,			#Smallest grid point
		   grid_max = 4.0,			#Largest grid point
		   grid_size = 200,			#Number of grid points
		   u = (c, γ = γ) -> isoelastic(c, γ),	#utility function
		   ∂u∂c = c -> c^(-γ),			#u'
		   f = k -> k^α,			#production function
		   f' = k -> α * k^(α - 1)		#f'
		   )


