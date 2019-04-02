using HDF5
using Distributed

include("datasets.jl")

@everywhere include("fisher_cont_meas.jl")
   
Ntime = 1000
QFI_unc_trajectories = 1000
FI_trajectories = 50000

h5open("fisher.h5", "w") do file
    for params in experimental_params
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