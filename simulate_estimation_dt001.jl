using Plots
pyplot()

using LaTeXStrings
using Statistics
#using StatsBase
using Random
using Printf

include("fluo_cont_meas_sim.jl")
include("likelihood_strong.jl")
include("fisher_cont_meas.jl")

# Experimental parameters
T1 = 15.0
TD = 5.0
Trabi = 2.
Tphi = 17.9

Ntrajectories = 500

# Parameter dictionary to be passed to the functions
params = Dict( :Tfinal    => 20., # Final time
               :dt        => 0.01, # duration of infinitesimal time
               :Gamma1    => 1. / T1,   # Gamma fluoresence
               :GammaD    => 1. / TD,  # Gamma dephasing controllable
               :GammaPhi  => 1. / Tphi,  # Gamma dephasing not controllable
               :etaF   => 0.14, # efficiency fluoresence heterodyne
               :etaD   => 0.34, # efficiency dephasing homodyne
               :omegaTrue => 2 * pi / Trabi, # True value of omega
               :omegaMin  => 2., # minimum value of omega
               :omegaMax  => 4., # maximum value of omega
               :threshold => 0.375,
               :unconditional_timesteps => 30,
               :Nomega    => 300); # Resolution in omega for the Bayesian estimation

println("Simulating trajectories...")
@time simData = parallel_fluo_continuous_measurement_het_simulation(Ntrajectories; params...) 

println("Estimating omega...")
@time simRes = likelihood_strong(simData.dyHet1, simData.dyHet2, simData.dyDep, simData.OutStrong, 2000; params...);

println("Evaluating Fisher...")
params_unconditional = copy(params)
params_unconditional[:etaF] = 0.
params_unconditional[:etaD] = 0.

@time fisherUncResult = parallel_fluo_continuous_measurement_het_classic_initial0(10; Ntime=1000, params_unconditional...)
@time fisherResult = parallel_fluo_continuous_measurement_het_classic_initial0(5000; Ntime=1000, params...);

println("Making plots...")
let res = simRes
    p1 = plot(res.t, res.omegaEst, ribbon=(res.sigmaBayes, res.sigmaBayes), fillalpha=0.15, label="Bayes")
    xlabel!("t")
    ylabel!("Omega")
    plot!(res.t, res.omegaMaxLik, label="Max-Lik")
    plot!(res.t, fill(params[:omegaTrue], size(res.t)), style=:dash, color=:black, label="True")
    title!(@sprintf "Estimated omega: %.4f" res.omegaEst[end])

    p2 = plot(fisherUncResult.t, fisherUncResult.QFisherEff, label="QFI uncond", legend=:topleft)
    plot!(fisherResult.t, fisherResult.QFisherEff, label="Eff QFI (monitoring + strong m.)")
    plot!(fisherResult.t, fisherResult.FisherAvg, label="FI monitoring")
    plot!(res.t, res.sigmaBayes .^ -2 ./ Ntrajectories, label="Inv. Bayesian variance (renorm.)")
    xlabel!("t")

    plot(p1, p2, layout = 2, size=(800,500))
    savefig("Simulate_estimation.png")
end