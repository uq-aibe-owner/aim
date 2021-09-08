#filepath cross system compatability code
if Sys.KERNEL === :NT || Sys.KERNEL === :Windows
    pathmark = "\\";
else
    pathmark = "/";
end

include("concordance.jl")

numSectors = 20
# for the 20 sector model
IOIGAs20=Array{Union{Nothing, String}}(nothing, length(IOIG));
for i in eachindex(IOIG);
    IOIGAs20[i] = IOIGTo20[IOIG[i]]
end

# reducing sectors by column
IOHalf = DataFrame(copy(transpose(IOSource[[4:117; 119; 121:126; 128], 3:116])),:auto);
insertcols!(IOHalf ,1, :Industry => IOIGAs20);
IOHalfSplitIndustry = groupby(IOHalf, :Industry);
IOHalf = combine(IOHalfSplitIndustry, valuecols(IOHalfSplitIndustry) .=> sum);

sort!(IOHalf)
IOHalf = select!(IOHalf, Not(:Industry));
IOHalf = transpose(Matrix(IOHalf[:,:]));

# reducing sectors by row
IO20 = DataFrame(copy([IOHalf[1:114, :] IOSource[4:117,117:126]]),:auto);
insertcols!(IO20 ,1, :Industry => IOIGAs20);
IO20SplitIndustry = groupby(IO20, :Industry);
IO20 = combine(IO20SplitIndustry, valuecols(IO20SplitIndustry) .=> sum);
IO20 = select!(IO20, Not(:Industry));
IO20 = Matrix(IO20[:,:])

reducedIO = [IO20; [IOHalf[115:122,:] IOSource[[119; 121:126; 128], 117:126]]];

reducedCapFlows = Matrix(select!(CSV.read("data"*pathmark*"capitalFlowsDraft.csv", DataFrame), Not(:Titles)));

# evaluating variables and parameters
# compensation of employees
LCurrent = zeros(numSectors)
# consumption?
CCurrent = zeros(numSectors)
# labour share
alpha = zeros(numSectors)
# intermediate share
mu = zeros(numSectors)
# consumption share
xi = zeros(numSectors)
# matrix of intermediate flows as shares
gammaM = zeros(numSectors,numSectors)
# matrix of capital investment as shares
gammaX = zeros(numSectors,numSectors)
# was called M, =reducedIO[numSectors+3,i]/(reducedIO[numSectors+1,i]+reducedIO[numSectors+3,i])


epsM = 0.1
epsQ = 1
epsD = 1
epsX = 1
epsLS = 1
deltaC = ones(numSectors)
# next ones are from aldrich
beta = 0.984
deltaK = 0.01

for i in 1:numSectors
    LCurrent[i] = reducedIO[numSectors+2,i]
    CCurrent[i] = reducedIO[numSectors+2,i] #fix this
    alpha[i] = reducedIO[numSectors+2,i]/(reducedIO[numSectors+2,i]+reducedIO[numSectors+3,i])
    mu[i] = reducedIO[numSectors+3,i]/(reducedIO[numSectors+1,i]+reducedIO[numSectors+3,i])
    xi[i] = (reducedIO[i,numSectors+2] + reducedIO[i,numSectors+3])/(reducedIO[numSectors+1,numSectors+2] + reducedIO[numSectors+1,numSectors+2])
    for j in 1:numSectors
        gammaM[i,j] = reducedIO[j,i]/(reducedIO[j,numSectors+1])
        gammaX[i,j] = reducedCapFlows[j,i]/sum(reducedCapFlows[:,i])
    end
end

#=
lagrange = E0*sum(beta*(log(sum(xi[i]^(1/epsD)*(deltaC[i]*C[i])^((epsD-1)/epsD))^(epsD/(epsD*0.999-1)))
-epsLS/(epsLS+1)+sum(L[i])^((epsLS+1)/epsLS)+sum(P*(X[i]+(1-deltaK)*K[i]-K[i]t+1))
+sum(P[i]*(Q[i]+(1-deltaC)*C[i]-C[i]t+1 - sum(M[i,j]+X[i,j])))))
=#


