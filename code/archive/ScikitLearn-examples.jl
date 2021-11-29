
using ScikitLearn
using ScikitLearn.GridSearch
using PyPlot
using GaussianProcesses: GPE, MeanZero, SE
using Random
import Random
Random.seed!(42)
# Training d:w
n = 10
x = 2π * rand(n, 1)
y = sin.(x[:, 1]) + 0.05*randn(n)

# Select mean and covariance function
mZero = MeanZero()                   # Zero mean function
kern = SE(0.0,0.0)                   # Squared exponential kernel with parameters
                                     # log(ℓ) = 0.0, log(σ) = 0.0
gp = fit!(GPE(m=mZero,k=kern, logNoise=-1.0), x,y);

gp_cv = fit!(GridSearchCV(GPE(m=mZero,k=SE(0.0,0.0)), Dict(:logNoise=>collect(-10:0.3:10), :k_lσ=>collect(0:0.1:5))), x, y);
best_gp = gp_cv.best_estimator_;
@show get_params(best_gp)[:logNoise] get_params(best_gp)[:k_lσ]
nothing

xx = -5:0.1:10
plot(xx, predict(gp, reshape(collect(xx), length(xx), 1)), label="hand-specified")
plot(xx, predict(best_gp, reshape(collect(xx), length(xx), 1)), label="gridsearch-optimized")
plot(x, y, "bo")
legend();

get_params(gp)
