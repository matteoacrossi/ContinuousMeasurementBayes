using Distributed
using Printf
#addprocs(2)

@everywhere using Pkg
@everywhere Pkg.activate(".")

using Plots
pyplot()

using LaTeXStrings
using Statistics
using StatsBase
using Random

@everywhere include("fluo_cont_meas_sim.jl")
@everywhere include("likelihood.jl")
@everywhere include("likelihood_strong.jl")
@everywhere include("fisher_cont_meas.jl")

@everywhere begin
    # Experimental parameters
    T1 = 15.0
    TD = 5.0
    Trabi = 2.
    Tphi = 17.9
    
    Ntrajectories = 500
    nomegas = 5
    dt = 0.1

    # Parameter dictionary to be passed to the functions
    params = Dict( :Tfinal    => 20., # Final time
                    :dt        => dt, # duration of infinitesimal time
                    :Gamma1    => 1. / T1,   # Gamma fluoresence
                    :GammaD    => 1. / TD,  # Gamma dephasing controllable
                    :GammaPhi  => 1. / Tphi,  # Gamma dephasing not controllable
                    :etaF   => 0.14, # efficiency fluoresence heterodyne
                    :etaD   => 0.34, # efficiency dephasing homodyne
                    :omegaTrue => 2 * pi / Trabi, # True value of omega
                    :omegaMin  => 2., # minimum value of omega
                    :omegaMax  => 4., # maximum value of omega
                    :threshold => 0.375,
                    :unconditional_timesteps => 3,
                    :Nomega    => 300); # Resolution in omega for the Bayesian estimation

end

    

@everywhere function vsomega(omega, Ntrajectories, initialparams)
    params = copy(initialparams)
    params[:omegaTrue] = omega
    params[:omegaMin] = max(0, params[:omegaTrue] - 1)
    params[:omegaMax] = params[:omegaTrue] + 1
    simData = parallel_fluo_continuous_measurement_het_simulation(Ntrajectories; params...) 
    @time simRes = likelihood_strong(simData; params...);
    return simRes.omegaEst[end], simRes.sigmaBayes[end]
end

# Basic linear regression
linreg(x, y) = reverse([x ones(length(x))] \ y)

omegas = rand(0.1:0.00001:5, nomegas)
@time res = pmap(o -> vsomega(o, Ntrajectories, params), omegas, batch_size=nprocs()-1)

println("Making plot...")
processed = collect.(collect(zip(res...)))
coeff = linreg(omegas, processed[1] - omegas)
errors = processed[2]
scatter(omegas, (processed[1] .- omegas), yerror=errors, label="Sim")
plot!([0,5],[0,0], style=:dash, color=:black, label=L"\Omega_{true}")
plot!([0,5], coeff[2] .* [0,5] .+ coeff[1], label= (@sprintf "Fit: %.3f x + %.3f" coeff[2] coeff[1]))
xlabel!(L"\Omega_{true}")
ylabel!(L"\Omega_{est}- \Omega_{true}")
title!(@sprintf "%d trajectories, dt %.2f" Ntrajectories dt)

savefig("Estimation_vs_omega.png")