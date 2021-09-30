using DataFrames, JuMP, Ipopt, DelimitedFiles;

include("reducedIO.jl")

# evaluating variables and parameters
#A = ones(numSectors)
# compensation of employees
#LCurrent = zeros(numSectors)
# consumption?
#CCurrent = zeros(numSectors)
# output
#QCurrent = zeros(numSectors)
#
#MCurrentT = zeros(numSectors)
#
#XCurrentT = zeros(numSectors)
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
#MCurrent = zeros(numSectors,numSectors)
# capital flows - sector to sector
#XCurrent = zeros(numSectors,numSectors)
# was called M, =reducedIO[numSectors+3,i]/(reducedIO[numSectors+1,i]+reducedIO[numSectors+3,i])
#KNext = zeros(numSectors)

epsM = 0.1*0.98
epsQ = 1*0.98
epsD = 1*0.98
epsX = 1*0.98
epsLS = 1*0.98
deltaC = ones(numSectors)*0.0098
# next ones are from aldrich
beta = 0.984*0.98
deltaK = 0.01*0.98

#calculating from data
for i in 1:numSectors
    #QCurrent[i] = reducedIO[numSectors+8,i]
    #LCurrent[i] = reducedIO[numSectors+2,i]
    #CCurrent[i] = (reducedIO[i,numSectors+2]+reducedIO[i,numSectors+3])
    alpha[i] = reducedIO[numSectors+2,i]/(reducedIO[numSectors+2,i]+reducedIO[numSectors+3,i])+0.001
    mu[i] = reducedIO[numSectors+3,i]/(reducedIO[numSectors+1,i]+reducedIO[numSectors+3,i])+0.001
    xi[i] = (reducedIO[i,numSectors+2] + reducedIO[i,numSectors+3])/(reducedIO[numSectors+1,numSectors+2] + reducedIO[numSectors+1,numSectors+2])+0.001
    capFlowsReceivable[i] = sum(reducedCapFlows[:,i])+0.001
    for j in 1:numSectors
        gammaM[i,j] = reducedIO[j,i]/(reducedIO[j,numSectors+1])+0.001
        gammaX[i,j] = reducedCapFlows[j,i]/sum(reducedCapFlows[:,i])+0.001
        #MCurrent[i,j] = reducedIO[i,j]
        #XCurrent[i,j] = reducedCapFlows[i,j]
    end
end

A = ones(numSectors).*0.98;


modBasic = Model(Ipopt.Optimizer)
@variable(modBasic, CCurrent[1:numSectors]>=0.01)
@variable(modBasic, LCurrent[1:numSectors]>=0.01)
@variable(modBasic, KCurrent[1:numSectors]>=0.01)
@variable(modBasic, KNext[1:numSectors]>=0.01)
@variable(modBasic, KNextNext[1:numSectors]>=0.01)
@variable(modBasic, MCurrent[1:numSectors,1:numSectors]>=0.01)
@variable(modBasic, XCurrent[1:numSectors,1:numSectors]>=0.01)
@variable(modBasic, QCurrent[1:numSectors]>=0.01)
@variable(modBasic, MCurrentT[1:numSectors]>=0.01)
@variable(modBasic, XCurrentT[1:numSectors]>=0.01)
@NLobjective(modBasic, Max, beta*(log(sum(xi[j]^(1/epsD)*(deltaC[j]*CCurrent[j])^((0.0001+epsD-1)/epsD) for j in 1:numSectors)^((0.0001+epsD)/(epsD*0.999-1)))
-epsLS/(0.999*epsLS+1)+sum(LCurrent[i] for i in 1:numSectors)^((epsLS+1)/0.999*epsLS)));
# Market clearing constraint
@NLconstraint(modBasic, sum(QCurrent[j]-CCurrent[j] - sum(MCurrent[i,j]+XCurrent[i,j] for i in 1:numSectors) for j in 1:numSectors) == 0);
@NLconstraint(modBasic, sum(XCurrentT[j]+(1-deltaK)*KCurrent[j]-KNext[j] for  j in 1:numSectors) == 0);
@NLconstraint(modBasic, sum((1-deltaK)*KNext[j]-KNextNext[j] for j in 1:numSectors) == 0);
for j in 1:numSectors;
    #@NLconstraint(modBasic, XCurrentT[j]==sum(gammaX[i,j]^(1/(0.0001+epsX))*XCurrent[i,j]^((0.0001+epsX-1)/epsX) for i in 1:numSectors)^((0.0001+epsX)/(0.0001+epsX-1)));
    #@NLconstraint(modBasic, MCurrentT[j]==sum(gammaM[i,j]^(1/epsM)*MCurrent[i,j]^((0.0001+epsM-1)/epsM) for i in 1:numSectors)^((0.0001+epsM)/(epsM-1)));
    #impose Q constraint from appendix F.1
    @NLconstraint(modBasic, A[j]*((1-mu[j])^(1/(0.0001+epsQ))*((KCurrent[j]/(0.0001+alpha[j]))^(0.0001+alpha[j])*(LCurrent[j]/(1-alpha[j]))^(1-alpha[j]))^((0.0001+epsQ-1)/epsQ)+mu[j]^(1/(0.0001+epsQ))*MCurrentT[j]^((0.0001+epsQ-1)/epsQ))^(epsQ/(0.99*epsQ-1)) == QCurrent[j]);
end
optimize!(modBasic)
#=
lagrange = E0*sum(beta*(log(sum(xi[j]^(1/epsD)*(deltaC[j]*C[j])^((epsD-1)/epsD))^(epsD/(epsD*0.999-1)))
-epsLS/(epsLS+1)+sum(LCurrent[j])^((epsLS+1)/epsLS)
+sum(P[j]*(QCurrent[j]-CCurrent[j] - sum(MCurrent[j,i]+XCurrent[j,i])))))
=#

#=
constraints:

sum(QCurrent[j]-CCurrent[j] - sum(MCurrent[j,i]+XCurrent[j,i])) == 0

sum(XCurrentT[j]+(1-deltaK)*KCurrent[j]-Kt+1[j]) == 0

FOC's:
P_t-1 - beta*P*(1-deltaC[j])=beta*xi[j]^(1/epsD)*deltaC[j]^((epsD-1)/epsD)*
CCurrent[j]^(-1/epsD)*(sum(xi[i]^(1/epsD)*(deltaC[i]*CCurrent[i])^((epsD-1)/eqsD))^(-1)

P[i]/P[j] = A[j]^((epsQ-1)/epsQ)*(QCurrent[j]*mu[j]/MCurrentOne[j])^(1/epsQ)*(MCurrentOne[j]*gammaM[j])^(1/epsM)

=#