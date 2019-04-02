include("do_estimation.jl")

@show ARGS
expid = Base.parse(Int64, ARGS[1])
do_estimation([10 100 1000], experimental_params[expid])