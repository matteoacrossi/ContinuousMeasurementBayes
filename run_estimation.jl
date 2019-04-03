include("do_estimation.jl")

@show ARGS
expid = Base.parse(Int64, ARGS[1])
do_estimation(vcat([50, 100], collect(500:500:10000)), experimental_params[expid])