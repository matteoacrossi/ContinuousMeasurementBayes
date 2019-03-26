using HDF5

# The data is organized so that 10k trajectories are measured along x, then 10k along y, then 10k along z and so on
CHUNK_SIZE = 10000
UNCONDITIONAL_STEPS = 3

"""
Load the data corresponding to the final z measurements.

Returns a named tuple containing the three currents (rescaled)
and the output of the strong measurement.
"""
function load_z_data(filename)
    file = h5open(filename)
    OutStrong = read(file["z"])
    chunk_n = Int(floor(length(OutStrong) / CHUNK_SIZE))
    indices = vcat([1 + (-1 + 3 * i ) * CHUNK_SIZE : 3* i * CHUNK_SIZE for i in 1:chunk_n]...)

    dyHet1 = read(file["u"])
    dyHet1 = prepend_unconditional(rescale_experimental_data.(dyHet1[:,indices]))

    dyHet2 = read(file["v"])
    dyHet2 = prepend_unconditional(rescale_experimental_data.(dyHet2[:,indices]))

    dyDep =  read(file["w"])
    dyDep = prepend_unconditional(rescale_experimental_data.(dyDep[:,indices]))

    return (dyHet1=dyHet1, dyHet2=dyHet2, dyDep=dyDep, OutStrong=prepend_unconditional(OutStrong))
end
 
"""
Get a random sample of Ntraj from the data tuple dataTuple
"""
function sample_data(Ntraj, dataTuple)
    idx = sample(1:length(dataTuple.OutStrong), Ntraj; replace=false, ordered=true)
    dyHet1 = dataTuple.dyHet1[:,idx]
    dyHet2 = dataTuple.dyHet2[:,idx]
    dyDep  = dataTuple.dyDep[:,idx]
    OutStrong = dataTuple.OutStrong[idx]
    return (dyHet1=dyHet1, dyHet2=dyHet2, dyDep=dyDep, OutStrong=OutStrong)
end

function rescale_experimental_data(x, factor=10^(-3/2))
    x .* factor
end

function prepend_unconditional(array)
    return vcat(zeros(eltype(array), UNCONDITIONAL_STEPS, size(array, 2)), array)
end

"""
Gets a sample chunk of trajectories starting from a random index
"""
function sample_data_chunk(Ntraj, filename)
    file = h5open(filename)
    u_data = file["u"] 
    v_data = file["v"]
    w_data = file["w"]
    z_data = file["z"]
    zidx = rand(1:(size(z_data,1)-Ntraj))
    idx = 2 * CHUNK_SIZE + Int(floor(zidx / CHUNK_SIZE)) * 3 * CHUNK_SIZE + 1
    dyHet1 = prepend_unconditional(u_data[:,idx:idx+Ntraj-1])
    dyHet2 = prepend_unconditional(v_data[:,idx:idx+Ntraj-1])
    dyDep  = prepend_unconditional(w_data[:,idx:idx+Ntraj-1])
    OutStrong   = prepend_unconditional(z_data[zidx:zidx+Ntraj-1, 1])
    return (dyHet1, dyHet2, dyDep, OutStrong)
end

function get_data_chunk(zidx, Ntraj, filename)
    file = h5open(filename)
    u_data = file["u"] 
    v_data = file["v"]
    w_data = file["w"]
    z_data = file["z"]
    idx = 2 * CHUNK_SIZE + Int(floor(zidx / CHUNK_SIZE)) * 3 * CHUNK_SIZE + 1
    dyHet1 = prepend_unconditional(u_data[:,idx:idx+Ntraj-1])
    dyHet2 = prepend_unconditional(v_data[:,idx:idx+Ntraj-1])
    dyDep  = prepend_unconditional(w_data[:,idx:idx+Ntraj-1])
    OutStrong   = prepend_unconditional(z_data[zidx:zidx+Ntraj-1, 1])
    return (dyHet1, dyHet2, dyDep, OutStrong)
end