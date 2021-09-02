#concordance dict between ANZSIC divisions (19 sectors) and various other industry classifications
using XLSX, ExcelReaders, DataFrames, Tables, JuMP, Ipopt, NamedArrays, DelimitedFiles, CSV;

#filepath cross system compatability code
if Sys.KERNEL === :NT || Sys.KERNEL === :Windows
    pathmark = "\\";
else
    pathmark = "/";
end

#19 Sector to 4 Sector
from19To4 = Dict("A"=> "Primary",
"B" => "Primary",
"C" => "Secondary",
"D" => "Secondary",
"E" => "Secondary",
"F" => "Tertiary",
"G" => "Tertiary",
"H" => "Tertiary",
"I" => "Tertiary",
"J" => "Tertiary",
"K" => "Tertiary",
"L" => "Tertiary",
"M" => "Tertiary",
"N" => "Tertiary",
"O" => "Tertiary",
"P" => "Tertiary",
"Q" => "Tertiary",
"R" => "Tertiary",
"S" => "Tertiary")

#IOIG to 19 Sector
IOSource = ExcelReaders.readxlsheet("data"*pathmark*"5209055001DO001_201819.xls", "Table 5");
IOIG = IOSource[4:117, 1];
ANZSICDiv = ["Agriculture, forestry and fishing", "Mining", "Manufacturing", "Electricity, gas, water and waste services", 
"Construction", "Wholesale trade", "Retail trade", "Accomodation and food services", 
"Transport, postal and warehousing", "Information media and telecommunications", "Financial and insurance services", 
"Rental, hiring and real estate services", "Professional, scientific and technical services", 
"Administrative and support services", "Public administration and safety", "Education and training", 
"Health care and social assistance", "Arts and recreation services", "Other services"];
ANZSICDivShort = ["AgrForestFish", "Mining", "Manufacturing", "Utilities", "Construction", "Wholesale", "Retail", "AccomFoodServ", 
"Transport&Ware", "Communications", "Finance&Insur", "RealEstate", "BusinessServ", "Admin", "PublicAdminSafe", 
"Education", "Health&Social", "Arts&Rec", "OtherServices"];
ANZSICDivByLetter =["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S"];
from19ToInd = Dict{String, Int64}();
for i in eachindex(ANZSICDivByLetter);
    from19ToInd[ANZSICDivByLetter[i]] = Int(i);
end
IOIGTo19 = Dict{Float64, String}();
for i in [1:1:114;];
    test = trunc(IOIG[i]/100);
    if 1 <= test <= 5
        IOIGTo19[IOIG[i]]="A"
    elseif 6 <= test <= 10;
        IOIGTo19[IOIG[i]]="B"
    elseif 11 <= test <= 25;
        IOIGTo19[IOIG[i]]="C"
    elseif 26 <= test <= 29;
        IOIGTo19[IOIG[i]]="D"
    elseif 30 <= test <= 32;
        IOIGTo19[IOIG[i]]="E"
    elseif 33 <= test <= 38;
        IOIGTo19[IOIG[i]]="F"
    elseif 39 <= test <= 43;
        IOIGTo19[IOIG[i]]="G"
    elseif 44 <= test <= 45;
        IOIGTo19[IOIG[i]]="H"
    elseif 46 <= test <= 53;
        IOIGTo19[IOIG[i]]="I"
    elseif 54 <= test <= 60;
        IOIGTo19[IOIG[i]]="J"
    elseif 62 <= test <= 64;
        IOIGTo19[IOIG[i]]="K"
    elseif 66 <= test <= 67;
        IOIGTo19[IOIG[i]]="L"
    elseif 69 <= test <= 70;
        IOIGTo19[IOIG[i]]="M"
    elseif 72 <= test <= 73;
        IOIGTo19[IOIG[i]]="N"
    elseif 75 <= test <= 77;
        IOIGTo19[IOIG[i]]="O"
    elseif 80 <= test <= 82;
        IOIGTo19[IOIG[i]]="P"
    elseif 84 <= test <= 87;
        IOIGTo19[IOIG[i]]="Q"
    elseif 89 <= test <= 92;
        IOIGTo19[IOIG[i]]="R"
    elseif 94 <= test <= 96;
        IOIGTo19[IOIG[i]]="S"
    else
        print("ERROR: An input has fallen outside of the range of categories")
    end
end

#ISIC 4.0 To 19 Sectors
ANZSICISICSource = CSV.read("data"*pathmark*"ANZSIC06-ISIC3pt1.csv", DataFrame);
ANZSIC19 = ANZSICISICSource[6:1484, 1][findall(x -> typeof(x)<:String, ANZSICISICSource[6:1484, 4])];
ISIC = ANZSICISICSource[6:1484, 4][findall(x -> typeof(x)<:String, ANZSICISICSource[6:1484, 4])];
for i in eachindex(ISIC);
    ISIC[i]=strip(ISIC[i], ['p']);
end
ISICTo19 = Dict(ISIC .=> ANZSIC19);

#NAIC2007 To 19 Sectors via ISIC 4.0
NAICSISICSource = ExcelReaders.readxlsheet("data"*pathmark*"2007_NAICS_to_ISIC_4.xls", "NAICS 07 to ISIC 4 technical");
NAICS = string.(Int.(NAICSISICSource[4:1768,1]));
ISICAsANZSIC = NAICSISICSource[4:1768,3];
ISICAsANZSIC = string.(ISICAsANZSIC);
containsX = findall( x -> occursin("X", x), ISICAsANZSIC);
ISICAsANZSIC[containsX] = replace.(ISICAsANZSIC[containsX], "X" => "1");
ISICAsANZSIC = parse.(Float64, ISICAsANZSIC);
NAICSANZSIC19 = string.(zeros(length(ISICAsANZSIC)));
for i in eachindex(ISICAsANZSIC);
    NAICSANZSIC19[i] = ISICTo19[lpad(Int(ISICAsANZSIC[i]),4,"0")];
end
NAICS07To19 = Dict(NAICS .=> NAICSANZSIC19);

#NAIC2002 To 19 Sectors via NAIC2007
NAICS02To07 = CSV.read("data"*pathmark*"2002_to_2007_NAICS.csv", DataFrame);
NAICS02To0702 = string.(NAICS02To07[3:1202, 1]);
NAICS02To0707 = string.(NAICS02To07[3:1202, 3]);
NAICS07As19 = string.(zeros(length(NAICS02To0707)));
for i in eachindex(NAICS02To0707);
    NAICS07As19[i] = NAICS07To19[NAICS02To0707[i]];
end
NAICS02To19 = Dict(NAICS02To0702 .=> NAICS07As19);

#NAIC1997 To 19 Sectors via NAIC2002
NAICS97To02 = CSV.read("data"*pathmark*"1997_NAICS_to_2002_NAICS.csv", DataFrame);
NAICS97To0297 = string.(NAICS97To02[1:1355, 1]);
NAICS97To0202 = string.(NAICS97To02[1:1355, 3]);
NAICS02As19 = string.(zeros(length(NAICS97To0202)));
for i in eachindex(NAICS97To0202);
    NAICS02As19[i] = NAICS02To19[NAICS97To0202[i]];
end
NAICS97To19 = Dict(NAICS97To0297 .=> NAICS02As19);
NAICS97To0297Trunc = first.(string.(NAICS97To02[1:1355, 1]),4);
NAICS97To19Trunc = Dict(NAICS97To0297Trunc .=> NAICS02As19);

#Comm180 To 19 Sectors via NAIC 1997
NAICS97ToComm180 = CSV.read("data"*pathmark*"NAICS_to_Comm180.csv", DataFrame);
NAICS97ToComm18097 = first.([NAICS97ToComm180[1:90,4];NAICS97ToComm180[1:89,9]],4);
containsStar = findall( x -> occursin("*", x), NAICS97ToComm18097);
NAICS97ToComm18097[containsStar] = replace.(NAICS97ToComm18097[containsStar], "*" => "");
tooShort = findall( x -> occursin(",", x), NAICS97ToComm18097);
NAICS97ToComm18097[tooShort] = first.(NAICS97ToComm18097[tooShort],2);
NAICS97ToComm180180 = [NAICS97ToComm180[1:90,2];NAICS97ToComm180[1:89,7]];
containsStar = findall( x -> occursin("*", x), NAICS97ToComm180180);
NAICS97ToComm180180[containsStar] = replace.(NAICS97ToComm180180[containsStar], "*" => "");
containsSpace = findall( x -> occursin(" ", x), NAICS97ToComm180180);
NAICS97ToComm180180[containsSpace] = replace.(NAICS97ToComm180180[containsSpace], " " => "");
NAICS97As19 = string.(zeros(length(NAICS97ToComm18097)));
for i in eachindex(NAICS97ToComm18097);
    NAICS97ToComm18097[i] = rpad(parse(Int64, NAICS97ToComm18097[i]),4,"1");
end
Invalid4Dig = findall( x -> occursin("2311", x), NAICS97ToComm18097);
NAICS97ToComm18097[Invalid4Dig].=["2331"];
for i in eachindex(NAICS97ToComm18097);
    NAICS97As19[i] = NAICS97To19Trunc[NAICS97ToComm18097[i]];
end
Comm180To19=Dict(NAICS97ToComm180180 .=> NAICS97As19);

#Final concordance
finalConcordance = [NAICS97ToComm180180 NAICS97As19];
writedlm("data"*pathmark*"Comm180To19Concordance.csv", finalConcordance, ',');
