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



