include("do_estimation.jl")

@show ARGS
expid = Base.parse(Int64, ARGS[1])
do_estimation(vcat(500, collect(1000:1000:10000)), experimental_params[expid], true)