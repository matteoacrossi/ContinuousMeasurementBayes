using JLD2

# Parameters
Ntrajectories = 10
Tfinal = 20. # Final time
dt = 0.1 # duration of infinitesimal time
Gamma1 = 1. / 15   # Gamma fluoresence
GammaD = 1. / 0.3   # Gamma dephasing controllable
GammaPhi = 1. / 17.9  # Gamma dephasing not controllable
etavalF = 0.14 # efficiency fluoresence heterodyne
etavalD = 0.34 # efficiency dephasing homodyne
omegaTrue = 2 * pi / 5. # True value of omega
omegaMin= 0. # minimum value of omega
omegaMax= 3. # maximum value of omega
Nomega = 200 # Resolution in omega for the Bayesian estimation

@save "parameters.jld"

include("fluo_cont_meas_sim.jl")
include("likelihood_all.jl")

@time (t, Ntime, dyHet1, dyHet2, dyDep, OutStrong, AvgZCondTrue) = parallel_fluo_continuous_measurement_het_simulation(Ntrajectories;
        Tfinal = Tfinal, # Final time
        dt = dt, # duration of infinitesimal time
        Gamma1 = Gamma1 ,   # Gamma fluoresence
        GammaD = GammaD,    # Gamma dephasing controllable
        GammaPhi = GammaPhi,  # Gamma dephasing not controllable
        etavalF=etavalF, #efficiency fluoresence heterodyne
        etavalD=etavalD, #efficiency dephasing homodyne
        omegay = omegaTrue) # omegaz

@time (t, omegay, AvgZcond, probBayes, probBayesTraj, omegaEst, omegaMaxLik, sigmaBayes)= LikelihooodTrajNew(Ntrajectories, Ntime;
    Tfinal = Tfinal, # Final time
    Gamma1 = Gamma1 ,   # Gamma fluoresence
    GammaD = GammaD,    # Gamma dephasing controllable
    GammaPhi = GammaPhi,  # Gamma dephasing not controllable
    etavalF = etavalF, #efficiency fluoresence heterodyne
    etavalD = etavalD, #efficiency dephasing homodyne
    Nomega = Nomega, # number of values of omega
    omegaTrue = omegaTrue, #true value of omega
    omegaMin = omegaMin,
    omegaMax = omegaMax,
    dyHet1 = dyHet1, #output current 1 heterodyne (simgax)
    dyHet2 = dyHet2, #output current 2 heterodyne (simgay)
    dyDep = dyDep,  #output current 3 homodyne (sigmaz)
    OutS = OutStrong ); # output strong final measurement

@save "all_data.jld" t Ntime dyHet1 dyHet2 dyDep OutStrong AvgZCondTrue omegay AvgZcond probBayes probBayesTraj omegaEst omegaMaxLik sigmaBayes
@save "data.jld" t Ntime omegay probBayes omegaEst omegaMaxLik sigmaBayes

# Test if the average values of z coincide
@assert isapprox(AvgZcond, AvgZCondTrue, rtol=dt^2,atol=dt^2)
