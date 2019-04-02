"""
This file contains function for interacting with the datasets
"""

using HDF5
using JSON
using StatsBase

datapath = "data/"

symb_dict(d::Dict) = Dict(Symbol(k) => v for (k, v) in d)
experimental_params = symb_dict.(JSON.parse(open(datapath * "params.json")))

function get_fisher(;Filename, kwargs...)
    fisherfile = h5open(datapath * "fisher.h5", "r")
    symb_dict(read(fisherfile[Filename]))
end

function get_fisher(params::Dict)
    return get_fisher(; params...)
end

# The data is organized so that 10k trajectories are measured along x, then 10k along y, then 10k along z and so on
CHUNK_SIZE = 10000
UNCONDITIONAL_STEPS = 3

"""
Load the data corresponding to the final z measurements.

Returns a named tuple containing the three currents (rescaled and preprocessed)
and the output of the strong measurement.
"""
function load_data(filename)
    file = h5open(datapath * filename)
    t = @elapsed OutStrong = read(file["z"])
    @info "Loaded strong output" t size(OutStrong)
    chunk_n = Int(floor(length(OutStrong) / CHUNK_SIZE))
    indices = vcat([1 + (-1 + 3 * i ) * CHUNK_SIZE : 3* i * CHUNK_SIZE for i in 1:chunk_n]...)

    t = @elapsed dyHet1 = read(file["u"])
    dyHet1 = prepend_unconditional(rescale_experimental_data.(dyHet1[:,indices]))
    @info "Loaded and preprocessed u current" t size(dyHet1)

    t = @elapsed dyHet2 = read(file["v"])
    dyHet2 = prepend_unconditional(rescale_experimental_data.(dyHet2[:,indices]))
    @info "Loaded and preprocessed v current" t size(dyHet2)

    t = @elapsed dyDep =  read(file["w"])
    dyDep = prepend_unconditional(rescale_experimental_data.(dyDep[:,indices]))
    @info "Loaded and preprocessed w current" t size(dyDep)

    return (dyHet1=dyHet1, dyHet2=dyHet2, dyDep=dyDep, OutStrong=prepend_unconditional(OutStrong))
end
 
"""
Get a random sample of Ntraj from the data tuple dataTuple
"""
function sample_data(Ntraj, dataTuple)
    @info "Length" size(dataTuple.OutStrong, 1)
    idx = sample(1:size(dataTuple.OutStrong, 1), Ntraj; replace=false, ordered=true)
    dyHet1 = dataTuple.dyHet1[:,idx]
    dyHet2 = dataTuple.dyHet2[:,idx]
    dyDep  = dataTuple.dyDep[:,idx]
    OutStrong = dataTuple.OutStrong[idx]
    return (dyHet1=dyHet1, dyHet2=dyHet2, dyDep=dyDep, OutStrong=OutStrong)
end

"""
Rescale experimental data for use with our functions
"""
function rescale_experimental_data(x, factor=10^(-3/2))
    x .* factor
end

"""
Prepend zeros to the currents data
"""
function prepend_unconditional(array)
    return vcat(zeros(eltype(array), UNCONDITIONAL_STEPS, size(array, 2)), array)
end
