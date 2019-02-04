using HDF5
using Statistics

FILENAME = "093_exp=1_prepared.h5"

file = h5open(FILENAME, "r")

u = read(file["u"])
v = read(file["v"])
w = read(file["w"])

rescaling_coefficient = -0.15 / mean(w, dims=2)[1] 

dyHet1 = rescaling_coefficient .* u
dyHet2 = rescaling_coefficient .* v
dyDep = rescaling_coefficient .* w

# Parameters
Ntrajectories = 20
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
Ntime = 197

t = dt * 1:Ntime
include("fluo_cont_meas_sim.jl")
include("likelihood_all.jl")

@time (t, omegay, AvgZcond, probBayes, probBayesTraj, omegaEst, omegaMaxLik, sigmaBayes)= LikelihooodTrajNew(Ntrajectories, Ntime;
    Tfinal = t[end], # Final time
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
    dyDep = dyDep)#,  #output current 3 homodyne (sigmaz)
    #OutS = OutStrong ); # output strong final measurement

    using PyPlot

figure()
plot(t, omegaEst, label="Bayes")
plot(t, omegaEst + sigmaBayes, "C0--")
plot(t, omegaEst - sigmaBayes, "C0--")
plot(t, omegaMaxLik, label="Max-Lik")
plot(t, fill(omegaTrue, size(t)), label="True")
legend()
#savefig("bayes.pdf")
#close()