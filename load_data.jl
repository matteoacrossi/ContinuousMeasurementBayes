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


### OLDER functions
# The data is organized so that 10k trajectories are measured along x, then 10k along y, then 10k along z and so on
CHUNK_SIZE = 10000

"""
Gets a sample of trajectories by chosing randomly (slow)
"""
function sample_data(Ntraj, file)
    zidx = sample(1:size(file["z"],1), Ntraj; replace=false, ordered=true)
    idx = 2 * CHUNK_SIZE + floor(zidx / CHUNK_SIZE) * 3 * CHUNK_SIZE
    dyHet1 = hcat([file["u"][:,i] for i in idx])
    dyHet2 = hcat([file["v"][:,i] for i in idx])
    dyDep  = hcat([file["w"][:,i] for i in idx])
    return (dyHet1, dyHet2, dyDep)
end

"""
Gets a sample chunk of trajectories starting from a random index (slow)
"""
function sample_data_chunk(Ntraj, file)
    u_data = file["u"] 
    v_data = file["v"]
    w_data = file["w"]
    z_data = file["z"]
    zidx = rand(1:(size(z_data,1)-Ntraj))
    idx = 2 * CHUNK_SIZE + Int(floor(zidx / CHUNK_SIZE)) * 3 * CHUNK_SIZE + 1
    dyHet1 = u_data[:,idx:idx+Ntraj-1]
    dyHet2 = v_data[:,idx:idx+Ntraj-1]
    dyDep  = w_data[:,idx:idx+Ntraj-1]
    OutZ   = z_data[zidx:zidx+Ntraj-1, 1]
    return (dyHet1, dyHet2, dyDep, OutZ)
end

function get_data_chunk(zidx, Ntraj, file)
    u_data = file["u"] 
    v_data = file["v"]
    w_data = file["w"]
    z_data = file["z"]
    idx = 2 * CHUNK_SIZE + Int(floor(zidx / CHUNK_SIZE)) * 3 * CHUNK_SIZE + 1
    dyHet1 = u_data[:,idx:idx+Ntraj-1]
    dyHet2 = v_data[:,idx:idx+Ntraj-1]
    dyDep  = w_data[:,idx:idx+Ntraj-1]
    OutZ   = z_data[zidx:zidx+Ntraj-1, 1]
    return (dyHet1, dyHet2, dyDep, OutZ)
end

"""
Gets a sample of trajectories by chosing randomly (slow)
"""
function sample_data(Ntraj)
    idx = sample(1:size(file["u"],2), Ntraj; replace=false, ordered=true)
    dyHet1 = hcat([file["u"][:,i] for i in idx])
    dyHet2 = hcat([file["v"][:,i] for i in idx])
    dyDep  = hcat([file["w"][:,i] for i in idx])
    return (dyHet1, dyHet2, dyDep)
end

"""
Gets a sample chunk of trajectories starting from a random index (slow)
"""
function sample_data_chunk(Ntraj)
    idx = rand(1:(size(file["u"],2)-Ntraj))
    dyHet1 = file["u"][:,idx:idx+Ntraj-1]
    dyHet2 = file["v"][:,idx:idx+Ntraj-1]
    dyDep  = file["w"][:,idx:idx+Ntraj-1]
    return (dyHet1, dyHet2, dyDep)
end
