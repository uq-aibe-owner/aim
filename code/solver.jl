using DataFrames, JuMP, Ipopt, DelimitedFiles;

include("reducedIO.jl")

# evaluating variables and parameters
A = ones(numSectors)
# compensation of employees
LCurrent = zeros(numSectors)
# consumption?
CCurrent = zeros(numSectors)
# output
QCurrent = zeros(numSectors)
#
MCurrentT = zeros(numSectors)
#
XCurrrentT = zeros(numSectors)
# new capital
capFlowsReceivable = zeros(numSectors)
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
# intermediate inputs
MCurrent = zeros(numSectors,numSectors)
# capital flows - sector to sector
XCurrent = zeros(numSectors,numSectors)
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

#calculating from data
for i in 1:numSectors
    QCurrent[i] = reducedIO[numSectors+8,i]
    LCurrent[i] = reducedIO[numSectors+2,i]
    CCurrent[i] = (reducedIO[i,numSectors+2]+reducedIO[i,numSectors+3])
    alpha[i] = reducedIO[numSectors+2,i]/(reducedIO[numSectors+2,i]+reducedIO[numSectors+3,i])
    mu[i] = reducedIO[numSectors+3,i]/(reducedIO[numSectors+1,i]+reducedIO[numSectors+3,i])
    xi[i] = (reducedIO[i,numSectors+2] + reducedIO[i,numSectors+3])/(reducedIO[numSectors+1,numSectors+2] + reducedIO[numSectors+1,numSectors+2])
    capFlowsReceivable[i] = sum(reducedCapFlows[:,i])
    for j in 1:numSectors
        gammaM[i,j] = reducedIO[j,i]/(reducedIO[j,numSectors+1])
        gammaX[i,j] = reducedCapFlows[j,i]/sum(reducedCapFlows[:,i])
        MCurrent[i,j] = reducedIO[i,j]
        XCurrent[i,j] = reducedCapFlows[i,j]
    end
    MCurrentT[i]=sum(reducedIO[:,i])
    MCurrentT[i]=sum(reducedCapFlows[:,i])
end


modBasic = Model(Ipopt.Optimizer)
@variable(modBasic, CCurrent)
@NLobjective(modBasic, Max, sum(beta*(log(sum(xi[j]^(1/epsD)*(deltaC[j]*CCurrent[j])^((epsD-1)/epsD))^(epsD/(epsD*0.999-1)))
-epsLS/(epsLS+1)+sum(LCurrent[j])^((epsLS+1)/epsLS))) for i in 1:numSectors, j in 1:numSectors));

#=
lagrange = E0*sum(beta*(log(sum(xi[j]^(1/epsD)*(deltaC[j]*C[j])^((epsD-1)/epsD))^(epsD/(epsD*0.999-1)))
-epsLS/(epsLS+1)+sum(LCurrent[j])^((epsLS+1)/epsLS)
+sum(P[j]*(QCurrent[j]-CCurrent[j] - sum(MCurrent[j,i]+XCurrent[j,i])))))
=#

#=
constraints:

sum(P[j]*(QCurrent[j]-CCurrent[j] - sum(MCurrent[j,i]+XCurrent[j,i]))) == 0

sum(XCurrentT[j]+(1-deltaK)*KCurrent[j]-Kt+1[j]) == 0

C




FOC's:
P_t-1 - beta*P*(1-deltaC[j])=beta*xi[j]^(1/epsD)*deltaC[j]^((epsD-1)/epsD)*
CCurrent[j]^(-1/epsD)*(sum(xi[i]^(1/epsD)*(deltaC[i]*CCurrent[i])^((epsD-1)/eqsD))^(-1)

P[i]/P[j] = A[j]^((epsQ-1)/epsQ)*(QCurrent[j]*mu[j]/MCurrentOne[j])^(1/epsQ)*(MCurrentOne[j]*gammaM[j])^(1/epsM)

=#