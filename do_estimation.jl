using JLD2
using Printf
using Plots 
using LaTeXStrings

include("datasets.jl")
include("likelihood_strong.jl")
include("fluo_cont_meas_sim.jl")
resultspath = "results/"
function do_estimation(Ntraj_array, experimental_params, simulate=false)

    estparams = Dict(
        :omegaMin  => max(0., experimental_params[:omegaTrue] - 1) # minimum value of omega
        :omegaMax  => experimental_params[:omegaTrue] + 1, # maximum value of omega
        :Nomega => 400)
    
    @info simulate
    params = merge(experimental_params, estparams)

    if !simulate
        @info "Loading data..."
        @time experimental_data = load_data(experimental_params[:Filename])
    end


    for Ntrajectories in Ntraj_array
        @info "" Ntrajectories

        if simulate
            @info "Simulating data..."
            @time expData = parallel_fluo_continuous_measurement_het_simulation(Ntrajectories; params...)
        else
            @info "Sampling data..."
            expData = sample_data(Ntrajectories, experimental_data)
        end
        
        @info "Estimating"
        @time expResult = likelihood_strong(expData; params...)
        
        filename = "estimation_Td_" * @sprintf("%.1f", params[:Td]) *
                    "_Tr_" * @sprintf("%2.1f", params[:Tr]) *
                    "_NTraj_$Ntrajectories"
        filename *= simulate ? "_sim.jld" : ".jld"

        @save resultspath * filename expResult params Ntrajectories
        @info "File saved." resultspath * filename 
    end
end

function plot_estimation(filename::String)
    @load filename expResult params Ntrajectories

    f = get_fisher(;params...)

    let res = expResult
        p1 = plot(f[:t], f[:QFI_unc], label="QFI uncond", style=:dash, legend=:topleft)
        plot!(f[:t], f[:QFI_eff], style=:dash, label="Eff QFI")
        plot!(f[:t],  f[:FI_monitoring], style=:dash, label="FI monit")
        plot!(f[:t], f[:FI_monitoring_and_strong], style=:dash, label="FI monit + strong m.")

        plot!(res.t, res.sigmaBayes .^ -2 ./ Ntrajectories, label="Inv. Bayes var (norm.)")
        plot!([17, 20], fill(1. / (Ntrajectories*(res.sigmaStrong[Ntrajectories]^2)), 2), line = (:steppre, :arrow, 1, 2), label="Inv. strong Bayes Var")

        xlabel!("t")

        p2 = plot( res.t, res.omegaEst, ribbon=(res.sigmaBayes, res.sigmaBayes), fillalpha=0.15, label="Bayes",legend=:bottomright)
        plot!(res.t, res.omegaMaxLik, label="Max-Lik")
        plot!(res.t, fill(params[:omegaTrue], size(res.t)), style=:dash,  color=:black, label="True")
        plot!([17, 20], fill(res.omegaEstStrong[end], 2), line = (:steppre, :arrow, 1, 2), fillalpha=0.15, label="Strong")
        xlabel!("t")
        plot(p1, p2, size=(1000,400), layout=2, title="T_d = $(params[:Td]), T_r = $(params[:Tr]), $Ntrajectories traj.")#latexstring("T_d = $(params[:Td])"))
        savefig(filename * ".pdf")
    end
end