# Experimental parameters
T1 = 15.0
TD = 5.0
Trabi = 2.
Tphi = 17.9

NTrajectories = 100

# Parameter dictionary to be passed to the functions
params = Dict( :Tfinal    => 20., # Final time
               :dt        => 0.1, # duration of infinitesimal time
               :Gamma1    => 1. / T1,   # Gamma fluoresence
               :GammaD    => 1. / TD,  # Gamma dephasing controllable
               :GammaPhi  => 1. / Tphi,  # Gamma dephasing not controllable
               :etavalF   => 0.14, # efficiency fluoresence heterodyne
               :etavalD   => 0.34, # efficiency dephasing homodyne
               :omegaTrue => 2 * pi / Trabi, # True value of omega
               :omegaMin  => 2., # minimum value of omega
               :omegaMax  => 4., # maximum value of omega
               :threshold => 0.375,
               :Nomega    => 500); # Resolution in omega for the Bayesian estimation


include("fluo_cont_meas_sim.jl")
include("likelihood_strong.jl")

@time simData = parallel_fluo_continuous_measurement_het_simulation(NTrajectories; params...) 
@time simRes = Likelihood_strong(simData.dyHet1, simData.dyHet2, simData.dyDep, simData.OutStrong, 200; 
        internalsteps = 1, params...);
#@save "all_data.jld" t Ntime dyHet1 dyHet2 dyDep OutStrong AvgZCondTrue omegay AvgZcond probBayes probBayesTraj omegaEst omegaMaxLik sigmaBayes
#@save "data.jld" t Ntime omegay probBayes omegaEst omegaMaxLik sigmaBayes

# Test if the average values of z coincide
@assert isapprox(simData.AvgZcondTrue, simRes.AvgZcond, rtol=params[:dt]^2,atol=params[:dt]^2)
