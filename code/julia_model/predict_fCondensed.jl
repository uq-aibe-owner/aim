function predict_f(gp::GPBase, x::AbstractMatrix; full_cov::Bool=false)
    size(x,1) == gp.dim || throw(ArgumentError("Gaussian Process object and input observations do not have consistent dimensions"))
    if full_cov
        return predict_full(gp, x)
    else
        ## Calculate prediction for each point independently
        μ = Array{eltype(x)}(undef, size(x,2))
        σ2 = similar(μ)
        for k in 1:size(x,2)
            m, sig = predict_full(gp, x[:,k:k])
            μ[k] = m[1]
            σ2[k] = max(diag(sig)[1], 0.0)
        end
        return μ, σ2
    end
end

predict_full(gp::GPA, xpred::AbstractMatrix, Q::Approx) = predictMVN(gp,xpred, gp.x, gp.y, gp.kernel, gp.mean, gp.v, gp.covstrat, Q)

function predictMVN(xpred::AbstractMatrix, xtrain::AbstractMatrix, ytrain::AbstractVector,
    kernel::Kernel, meanf::Mean, alpha::AbstractVector,
    covstrat::CovarianceStrategy, Ktrain::AbstractPDMat)
crossdata = KernelData(kernel, xtrain, xpred)
priordata = KernelData(kernel, xpred, xpred)
Kcross = cov(kernel, xtrain, xpred, crossdata)
Kpred = cov(kernel, xpred, xpred, priordata)
mx = mean(meanf, xpred)
mu, Sigma_raw = predictMVN!(Kpred, Ktrain, Kcross, mx, alpha)
return mu, Sigma_raw
end

function predictMVN!(Kxx, Kff, Kfx, mx, αf)
    mu = mx + Kfx' * αf
    Lck = whiten!(Kff, Kfx)
    subtract_Lck!(Kxx, Lck)
    return mu, Kxx
end

@inline function subtract_Lck!(Sigma_raw::AbstractArray{<:AbstractFloat}, Lck::AbstractArray{<:AbstractFloat})
    LinearAlgebra.BLAS.syrk!('U', 'T', -1.0, Lck, 1.0, Sigma_raw)
    LinearAlgebra.copytri!(Sigma_raw, 'U')
end