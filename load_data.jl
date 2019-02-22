using HDF5

"""
Load the data corresponding to the final z measurements.

Returns a named tuple containing the three currents (rescaled)
and the output of the strong measurement.
"""
function load_z_data(file)
    OutZ = read(file["z"])
    CHUNK_SIZE = 10000
    chunk_n = Int(floor(length(OutZ) / CHUNK_SIZE))
    indices = vcat([1 + (-1 + 3 * i ) * CHUNK_SIZE : 3* i * CHUNK_SIZE for i in 1:chunk_n]...)
    print(size(OutZ))
    dyHet1 = read(file["u"])
    dyHet1 = rescale_experimental_data.(dyHet1[:,indices])
    print(size(dyHet1))
    dyHet2 = read(file["v"])
    dyHet2 = rescale_experimental_data.(dyHet2[:,indices])
    print(size(dyHet2))
    dyDep =  read(file["w"])
    dyDep = rescale_experimental_data.(dyDep[:,indices])
    print(size(dyDep))            
    return (dyHet1=dyHet1, dyHet2=dyHet2, dyDep=dyDep, OutZ=OutZ)
end

function sample_data(Ntraj, dataTuple)
    idx = sample(1:length(dataTuple.OutZ), Ntraj; replace=false, ordered=true)
    dyHet1 = dataTuple.dyHet1[:,idx]
    dyHet2 = dataTuple.dyHet2[:,idx]
    dyDep  = dataTuple.dyDep[:,idx]
    OutZ = dataTuple.OutZ[idx]
    return (dyHet1=dyHet1, dyHet2=dyHet2, dyDep=dyDep, OutZ=OutZ)
end

function rescale_experimental_data(x, factor=10^(-3/2))
    x .* factor
end