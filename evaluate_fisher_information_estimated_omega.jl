"""
This file evaluates the various QFIs for the avaliable parameter sets, and stores it into an HDF5 file.

It can be run on qtech using the batch.sh file. 
Parallel evaluation is available (specify number of cores)

    sbatch -c 10 batch.sh evaluate_fisher_information.jl

"""

using HDF5
using Distributed

include("datasets.jl")
include("likelihood_strong.jl")
@everywhere include("fisher_cont_meas.jl")
   
Ntrajectories = 10000

Ntime = 2000
QFI_unc_trajectories = 1000
FI_trajectories = 50000

h5open("data/fisher_est_omega.h5", "w") do file
    for params in experimental_params

        estparams = Dict(
            :omegaMin  => max(0., params[:omegaTrue] - 1), # minimum value of omega
            :omegaMax  => params[:omegaTrue] + 1, # maximum value of omega
            :Nomega => 400)

        params = merge(params, estparams)

        @info "Loading data"
        @time experimental_data = load_data(params[:Filename], peakfilter)
        expData = sample_data(Ntrajectories, experimental_data)

        @info "Estimating"
        @time expResult = likelihood_strong(expData; params...)

        @info "Initial omegaTrue was $(params[:omegaTrue])"
        params[:omegaTrue] = expResult.omegaEst[end]
        @info params[:omegaTrue]

        g = g_create(file, params[:Filename])
        # For the unconditional dynamics, we set zero efficiency
        params_unconditional = copy(params)
        params_unconditional[:etaF] = 0.
        params_unconditional[:etaD] = 0.

        @time fisherUncResult = parallel_fluo_continuous_measurement_het_classic_initial0(
            QFI_unc_trajectories; 
            Ntime=Ntime, params_unconditional...)
        @time fisherResult = parallel_fluo_continuous_measurement_het_classic_initial0(FI_trajectories; 
        Ntime=Ntime, params...);

        g["t"] = collect(fisherUncResult.t)
        g["QFI_unc"] = fisherUncResult.QFisherEff
        g["QFI_eff"] = fisherResult.QFisherEff
        g["FI_monitoring"] = fisherResult.FisherAvg
        g["FI_monitoring_and_strong"] = fisherResult.FisherMEff
        for (k, v) in params
            attrs(g)[String(k)] = v
        end
    end
end