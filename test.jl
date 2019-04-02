
Ntrajectories = 100

include("datasets.jl")
include("fluo_cont_meas_sim.jl")
include("likelihood_strong.jl")

estparams = Dict(
        :omegaMin  => 0., # minimum value of omega
        :omegaMax  => 5., # maximum value of omega
        :Nomega => 200)

params = merge(experimental_params[1], estparams)

@show params

@time simData = parallel_fluo_continuous_measurement_het_simulation(Ntrajectories; params...) 
@time simRes = likelihood_strong(simData; 
        params...);
        
# Test if the average values of z coincide
@assert isapprox(simData.AvgZcondTrue, simRes.AvgZcond, rtol=params[:dt]^2,atol=params[:dt]^2)


println("Tests passed!")