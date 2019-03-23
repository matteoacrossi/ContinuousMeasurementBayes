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

#@assert size(dyHet1) == size(dyHet2) == size(dyDep), "Current sizes don't match"

dimJ = 2; # Dimension of the corresponding Hilbert space

dt = Tfinal / Ntime;
    
Ntraj = size(dyHet1, 2)
    
omegay = collect(range(omegaMin, stop=omegaMax, length=Nomega))
jomegaTrue = findmin(abs.(omegay .- omegaTrue))[2]

priorG = PriorGaussian(omegay, 0., 1. * pi)

# Pauli operators
sx::Array{ComplexF64} = [0 1 ; 1 0] 
sy::Array{ComplexF64} = [0 -1im ; 1im 0]
sz::Array{ComplexF64} = [1 0 ; 0 -1]

sm::Array{ComplexF64} = [0 0 ; 1 0]
sp::Array{ComplexF64} = [0 1 ; 0 0]

sx2 = sx^2
sy2 = sy^2
sz2 = sz^2

# We use static arrays to improve performance
cF = SMatrix{2,2}(sqrt(Gamma1) * sm)
cD = SMatrix{2,2}(sqrt(GammaD/2) * sz)
cPhi = SMatrix{2,2}(sqrt(GammaPhi/2) * sz)

# omega is the parameter that we want to estimate

PiPlus=SMatrix{2,2}([1 0 ; 0 0] .+ 0.0im)

# initial state of the system
RhoIn::Array{ComplexF64} = [0.04 0 ; 0 0.96];

# POVM operators for strong measurement
phi=0.; #phase of the measurement
Pi1=[cos(phi)^2 sin(phi)*cos(phi) ; sin(phi)*cos(phi) sin(phi)^2]

#
t = (1:Ntime)*dt;

rho = Array{ComplexF64}(undef, 2,2,Nomega);

probBayes = Array{Float64}(undef, Nomega, Ntime);    
lklhood = ones(Nomega)/Nomega

probStrong = Array{Float64}(undef, Nomega);    
lklhoodStrong = ones(Nomega)/Nomega

probBayesTraj = Array{Float64}(undef, Nomega, Ntraj,Ntime);    
lklhoodTraj = Array{Float64}(undef, Nomega);

omegaEst = Array{Float64}(undef, Ntime);
sigmaBayes = Array{Float64}(undef, Ntime);
omegaMaxLik = Array{Float64}(undef, Ntime);

omegaEstStrong = Array{Float64}(undef, Ntraj);
sigmaStrong = Array{Float64}(undef, Ntraj);
omegaMaxLikStrong = Array{Float64}(undef, Ntraj);
                                
AvgZcond = Array{Float64}(undef, Ntraj,Ntime)

H = [(o/2.) * sy for o in omegay]; # Hamiltonian of the qubit

M0 = I - ((cF'*cF)/2 + (cD'*cD)/2 + (cPhi'*cPhi)/2) * dt
rho0 = cat([RhoIn for i=1:Nomega]..., dims=3)

for ktraj = 1:Ntraj
    copy!(rho, rho0)
    for jt=1:Ntime

        if jt <= unconditional_timesteps
            M1 = M0;
        else
            M1 = M0 + sqrt(etavalF/2) * cF * (dyHet1[end - Ntime + jt, ktraj] - 
                                    1im * dyHet2[end - Ntime + jt, ktraj]) + 
                    sqrt(etavalD) * (cD * dyDep[end - Ntime + jt, ktraj])
        end
            
        for jomega = 1:Nomega
        #in questo ciclo mi calcolo le likelihood della misura al tempo jt per ciascun valore di omega
            
            #H = (omegay[jomega]/2.) * sy; # Hamiltonian of the qubit

            M = M1 - 1im * SMatrix{2,2}(H[jomega]) * dt
            rhotmp = SMatrix{2,2}(view(rho,:,:,jomega))

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
            
            lklhood[jomega] = real(tr(newRho));
            rho[:,:,jomega] = newRho / lklhood[jomega];
            
            lklhood[jomega] = lklhood[jomega] - (omegay[jomega] * dt / 2)^2;
                
              

            if jt == Ntime
                pPlus = real(tr(rho[:,:,jomega]*PiPlus));
                if OutZ[ktraj] >= threshold
                    lklhoodStrong[jomega] = pPlus;
                else
                    lklhoodStrong[jomega] = (1 - pPlus);
                end
            end
        end  # fine ciclo su omega
        
        # Chech for the mean value of z operator
        AvgZcond[ktraj,jt] = real(tr(rho[:,:,jomegaTrue]*sz));

        if jt == 1
            lklhoodTraj = lklhood  #likelihood singola traiettoria
            if ktraj == 1
                lklhood = lklhoodTraj#.*priorG[:];
            else
                lklhood = lklhood .* probBayes[:,jt];
                # se non è la prima traiettoria, all'inizio della dinamica devo moltiplicare per la 
                # probabilità a quel tempo calcolata per le traiettorie precedenti
            end
        else
            lklhoodTraj = lklhood .* probBayesTraj[:,ktraj,jt-1];
            if ktraj == 1
                lklhood = lklhoodTraj;
                # se è la prima traiettoria devo solo moltiplicare la likelihood per 
                # la probabilità fino a quel punto della dinamica
            else 
                lklhood  = lklhood .* probBayesTraj[:,ktraj,jt-1].*probBayes[:,jt];
                # moltiplico likelihood per probabilità traiettoria fino al tempo antecedente 
                # e per probabilità a quel tempo considerate tutte le traiettorie 
            end
        end
        

        norm = sum(lklhood);
        normTraj = sum(lklhoodTraj);
            
        probBayes[:,jt] = lklhood/norm;   # probabilità Bayesiana considerando tutte le traiettorie
        probBayesTraj[:,ktraj,jt]=lklhoodTraj /normTraj;  #probabilità Bayesiana singola traiettoria
        
                                                            
        if jt == Ntime
            if ktraj == 1
                lklhoodStrong = lklhoodStrong .* probBayesTraj[:,ktraj,Ntime]; 
            else
                lklhoodStrong = lklhoodStrong .* probBayesTraj[:,ktraj,Ntime] .* probStrong[:];
            end
            normStrong = sum(lklhoodStrong);
            probStrong[:] = lklhoodStrong/normStrong;
            
            omegaEstStrong[ktraj] = sum(probStrong[:].*omegay);
            sigmaStrong[ktraj] = sqrt(sum(probStrong[:].*(omegay.^2)) - omegaEstStrong[ktraj]^2);
                    
            indMStrong = argmax(probStrong[:]);   
            omegaMaxLikStrong[ktraj] = omegay[indMStrong];
        end
        
        
        if ktraj == Ntraj        
            omegaEst[jt] = sum(probBayes[:,jt].*omegay);
            sigmaBayes[jt] = sqrt(sum(probBayes[:,jt].*(omegay.^2)) - omegaEst[jt]^2)
                
            indM = argmax(probBayes[:,jt]);   
            omegaMaxLik[jt]=omegay[indM];
        end

    end #fine ciclo sul tempo
        
end # fine ciclo su traiettorie

return (t=t, 
        omegas=omegay, 
        AvgZcond=AvgZcond, 
        #probBayes=probBayes, 
        #probBayesTraj=probBayesTraj, 
        omegaEst=omegaEst, 
        omegaMaxLik=omegaMaxLik,
        sigmaBayes=sigmaBayes,
        omegaEstStrong=omegaEstStrong, 
        omegaMaxLikStrong=omegaMaxLikStrong, 
        sigmaStrong=sigmaStrong)
end