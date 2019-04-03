include("do_estimation.jl")

@show ARGS
expid = Base.parse(Int64, ARGS[1])
do_estimation([50 100 500 750 1000 2500 5000 10000], experimental_params[expid])