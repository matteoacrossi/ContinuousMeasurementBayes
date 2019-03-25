using LinearAlgebra
using StaticArrays

PriorGaussian(omega, omegaMean, Sigma) = exp.( - ((omega .- omegaMean).^2)/(2*Sigma^2))

function Likelihood_strong(dyHet1, dyHet2, dyDep, OutZ, Ntime;
    Tfinal = nothing, # Final time
    Gamma1 = nothing,   # Gamma fluoresence
    GammaD = nothing,    # Gamma dephasing controllable
    GammaPhi = nothing,  # Gamma dephasing not controllable
    etavalF = nothing, #efficiency fluoresence heterodyne
    etavalD = nothing, #efficiency dephasing homodyne
    omegaMin = nothing, #minimum value of omega
    omegaMax = nothing, #maximum value of omega
    Nomega = nothing, # number of values of omega
    omegaTrue = nothing, 
    unconditional_timesteps = nothing,
    threshold = nothing, kwargs...)  #true value of omega

@assert size(dyHet1) == size(dyHet2) == size(dyDep) "Current sizes don't match"
@assert ndims(OutZ) == 1 "OutZ should be monodimensional"
@assert length(OutZ) == size(dyHet1)[2] "Length of OutZ doesn't match other currents"

dimJ = 2; # Dimension of the Hilbert space

dt = Tfinal / Ntime
Ntraj = size(dyHet1, 2)
    
omegas = collect(range(omegaMin, stop=omegaMax, length=Nomega))
jomegaTrue = findmin(abs.(omegas .- omegaTrue))[2]

priorG = PriorGaussian(omegas, 0., 1. * pi)

# Pauli operators
sx::Array{ComplexF64} = [0 1 ; 1 0] 
sy::Array{ComplexF64} = [0 -1im ; 1im 0]
sz::Array{ComplexF64} = [1 0 ; 0 -1]

sm::Array{ComplexF64} = [0 0 ; 1 0]
sp::Array{ComplexF64} = [0 1 ; 0 0]

sx2 = sx^2
sy2 = sy^2
sz2 = sz^2

# Interaction operators
# We use static arrays to improve performance
cF = SMatrix{2,2}(sqrt(Gamma1) * sm)
cD = SMatrix{2,2}(sqrt(GammaD/2) * sz)
cPhi = SMatrix{2,2}(sqrt(GammaPhi/2) * sz)


PiPlus=SMatrix{2,2}([1 0 ; 0 0] .+ 0.0im)

# initial state of the system
RhoIn::Array{ComplexF64} = [0.04 0 ; 0 0.96]
# POVM operators for strong measurement
phi=0.; #phase of the measurement
Pi1=[cos(phi)^2 sin(phi)*cos(phi) ; sin(phi)*cos(phi) sin(phi)^2]

# time
t = (1:Ntime)*dt
# Array prealloaction
rho = Array{ComplexF64}(undef, 2, 2, Nomega)
probBayes = Array{Float64}(undef, Nomega, Ntime)
lklhood = ones(Nomega)/Nomega

probStrong = Array{Float64}(undef, Nomega)
lklhoodStrong = ones(Nomega)/Nomega

probBayesTraj = Array{Float64}(undef, Nomega, Ntraj, Ntime)
lklhoodTraj = Array{Float64}(undef, Nomega)
omegaEst = Array{Float64}(undef, Ntime)
sigmaBayes = Array{Float64}(undef, Ntime)
omegaMaxLik = Array{Float64}(undef, Ntime)
omegaEstStrong = Array{Float64}(undef, Ntraj)
sigmaStrong = Array{Float64}(undef, Ntraj)
omegaMaxLikStrong = Array{Float64}(undef, Ntraj)
AvgZcond = Array{Float64}(undef, Ntraj, Ntime)

# Hamiltonian of the qubit for all omegas
H = [(o/2.) * sy for o in omegas]

# Trajectory-independent part of the Kraus operator
M0 = I - ((cF'*cF)/2 + (cD'*cD)/2 + (cPhi'*cPhi)/2) * dt

rho0 = cat([RhoIn for i = 1:Nomega]..., dims = 3)

for ktraj = 1:Ntraj
    copy!(rho, rho0)
    for jt=1:Ntime

        if jt <= unconditional_timesteps
            M1 = M0
        else
            # Omega-independent part of the Kraus operator
            M1 = M0 + sqrt(etavalF/2) * cF * (dyHet1[end - Ntime + jt, ktraj] - 
                                    1im * dyHet2[end - Ntime + jt, ktraj]) + 
                    sqrt(etavalD) * (cD * dyDep[end - Ntime + jt, ktraj])
        end
            
        for jomega = 1 : Nomega
        #in questo ciclo mi calcolo le likelihood della misura al tempo jt per ciascun valore di omega
            
            #H = (omegas[jomega]/2.) * sy; # Hamiltonian of the qubit

            M = M1 - 1im * SMatrix{2,2}(H[jomega]) * dt
            
            rhotmp = SMatrix{2,2}(view(rho,:,:,jomega))
            
            # We update the density operator using Rouchon's formula
            newRho = M * rhotmp * M'
            if  jt <= unconditional_timesteps
                newRho += (dt * (cF * rhotmp * cF') +
                          dt * (cD * rhotmp * cD') +
                          dt * (cPhi * rhotmp * cPhi'))
            else        
                newRho += (dt * (1 - etavalF) * (cF * rhotmp * cF') + 
                          dt * (1 - etavalD) * (cD * rhotmp * cD') +  
                          dt * (cPhi * rhotmp * cPhi'))
            end
            
            # The likelihood is the trace of the density operator
            lklhood[jomega] = real(tr(newRho))
            rho[:,:,jomega] = newRho / lklhood[jomega]

            # We apply a correction corresponding to the effect of
            # the Hamiltonian on the trace 
            lklhood[jomega] = lklhood[jomega] - (omegas[jomega] * dt / 2)^2

            # At the end of the trajectory we perform the final 
            # strong measurement
            if jt == Ntime
                pPlus = real(tr(rho[:,:,jomega]*PiPlus))
                if OutZ[ktraj] >= threshold
                    lklhoodStrong[jomega] = pPlus
                else
                    lklhoodStrong[jomega] = (1 - pPlus)
                end
            end
        end  # end of cicle over omega
        
        # Chech for the mean value of z operator
        AvgZcond[ktraj,jt] = real(tr(rho[:,:,jomegaTrue]*sz))

        # We update the likelihood
        # TODO: Make code more readable
        if jt == 1
            lklhoodTraj = lklhood  #likelihood singola traiettoria
            if ktraj == 1
                lklhood = lklhoodTraj#.*priorG[:]
            else
                lklhood = lklhood .* probBayes[:,jt]
                # se non è la prima traiettoria, all'inizio della dinamica devo moltiplicare per la 
                # probabilità a quel tempo calcolata per le traiettorie precedenti
            end
        else
            lklhoodTraj = lklhood .* probBayesTraj[:,ktraj,jt-1]
            if ktraj == 1
                lklhood = lklhoodTraj
                # se è la prima traiettoria devo solo moltiplicare la likelihood per 
                # la probabilità fino a quel punto della dinamica
            else 
                lklhood  = lklhood .* probBayesTraj[:,ktraj,jt-1].*probBayes[:,jt]
                # moltiplico likelihood per probabilità traiettoria fino al tempo antecedente 
                # e per probabilità a quel tempo considerate tutte le traiettorie 
            end
        end
        
        # We normalize the likelihood and update the 
        # bayes probability
        norm = sum(lklhood)
        normTraj = sum(lklhoodTraj)
        # Bayesian probability considering all the trajectories
        probBayes[:, jt] = lklhood / norm   
        # Bayesian probability for the single trajectories
        probBayesTraj[:, ktraj, jt] = lklhoodTraj / normTraj  
        
        # We update the Bayesian probability for the final strong measurement
        if jt == Ntime
            if ktraj == 1
                lklhoodStrong = lklhoodStrong .* probBayesTraj[:,ktraj,Ntime]
            else
                lklhoodStrong = lklhoodStrong .* probBayesTraj[:,ktraj,Ntime] .* probStrong
            end
            normStrong = sum(lklhoodStrong)
            probStrong = lklhoodStrong/normStrong
            omegaEstStrong[ktraj] = sum(probStrong .* omegas)
            sigmaStrong[ktraj] = sqrt(sum(probStrong .* (omegas .^ 2)) - omegaEstStrong[ktraj]^2)
            indMStrong = argmax(probStrong)
            omegaMaxLikStrong[ktraj] = omegas[indMStrong]
        end
        
        if ktraj == Ntraj        
            omegaEst[jt] = sum(probBayes[:,jt].*omegas)
            sigmaBayes[jt] = sqrt(sum(probBayes[:,jt].*(omegas.^2)) - omegaEst[jt]^2)
                
            indM = argmax(probBayes[:,jt])
            omegaMaxLik[jt]=omegas[indM]
        end

    end # end cycle on time
        
end # end cycle on trajectories

return (t=t, 
        omegas=omegas, 
        AvgZcond=AvgZcond, 
        probBayes=probBayes, 
        probBayesTraj=probBayesTraj, 
        omegaEst=omegaEst, 
        omegaMaxLik=omegaMaxLik,
        sigmaBayes=sigmaBayes,
        omegaEstStrong=omegaEstStrong, 
        omegaMaxLikStrong=omegaMaxLikStrong, 
        sigmaStrong=sigmaStrong)
end