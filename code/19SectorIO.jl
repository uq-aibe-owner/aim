#filepath cross system compatability code
if Sys.KERNEL === :NT || Sys.KERNEL === :Windows
    pathmark = "\\";
else
    pathmark = "/";
end

include("concordance.jl")

IOIGAs19=Array{Union{Nothing, String}}(nothing, length(IOIG));
for i in eachindex(IOIG);
    IOIGAs19[i] = IOIGTo19[IOIG[i]]
end

IOHalf = DataFrame(copy(transpose(IOSource[[4:117; 119; 121:126; 128], 3:116])),:auto);
insertcols!(IOHalf ,1, :Industry => IOIGAs19);
IOHalfSplitIndustry = groupby(IOHalf, :Industry);
IOHalf = combine(IOHalfSplitIndustry, valuecols(IOHalfSplitIndustry) .=> sum);

sort!(IOHalf)
IOHalf = select!(IOHalf, Not(:Industry));
IOHalf = transpose(Matrix(IOHalf[:,:]));

IO19 = DataFrame(copy([IOHalf[1:114, :] IOSource[4:117,117:126]]),:auto);
insertcols!(IO19 ,1, :Industry => IOIGAs19);
IO19SplitIndustry = groupby(IO19, :Industry);
IO19 = combine(IO19SplitIndustry, valuecols(IO19SplitIndustry) .=> sum);
IO19 = select!(IO19, Not(:Industry));
IO19 = Matrix(IO19[:,:])

fullIO19 = [IO19; [IOHalf[115:122,:] IOSource[[119; 121:126; 128], 117:126]]];

paramLtJ = zeros(19)
paramAlphaJ = zeros(19)
paramMtJ = zeros(19)
paramMuJ = zeros(19)
xiJ = zeros(19)
for i in 1:19
    paramLtJ[i] = fullIO19[21,i]
    paramAlphaJ[i] = fullIO19[21,i]/(fullIO19[21,i]+fullIO19[22,i])
    paramMtJ[i] = fullIO19[20,i]/(fullIO19[20,i]+fullIO19[22,i])
    paramMuJ[i] = fullIO19[22,i]/(fullIO19[20,i]+fullIO19[22,i])
    xiJ[i] = (fullIO19[i,21] + fullIO19[i,22])/(fullIO19[20,21] + fullIO19[20,22])
end