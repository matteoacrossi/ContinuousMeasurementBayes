PriorGaussian(omega,omegaMean,Sigma)=exp.(-((omega .- omegaMean).^2)/(2*Sigma^2))

# in questa funzione faccio stima bayesiana sul tutte le traiettorie
function LikelihooodTrajNew(Ntraj, Ntime;
    Tfinal = 10., # Final time
    Gamma1 = 1. / 15 ,   # Gamma fluoresence
    GammaD = 1. / (0.3),    # Gamma dephasing controllable
    GammaPhi = 1. / (17.9),  # Gamma dephasing not controllable
    etavalF=0.14, #efficiency fluoresence heterodyne
    etavalD=0.34, #efficiency dephasing homodyne
    omegaMin = -1. *pi, #minimum value of omega
    omegaMax=1. *pi, #maximum value of omega
    Nomega = 500, # number of values of omega
    omegaTrue = 2. *pi/5.,  #true value of omega
    dyHet1 = zeros(Ntime,Ntraj), #output current 1 heterodyne (simgax)
    dyHet2 = zeros(Ntime,Ntraj), #output current 2 heterodyne (simgay)
    dyDep = zeros(Ntime,Ntraj),  #output current 3 homodyne (sigmaz)
    OutS = zeros(Ntime,Ntraj) ) # output strong final measurement

dimJ = 2; # Dimension of the corresponding Hilbert space

dt = Tfinal / Ntime;

domega = (omegaMax-omegaMin)/Nomega;

omegay = Array{Float64}(undef, Nomega+1);  
priorG = Array{Float64}(undef, Nomega+1); 

for jomega=1:(Nomega+1)
omegay[jomega] = real(omegaMin + (jomega-1)*domega);
priorG[jomega] = real(PriorGaussian(omegay[jomega],0.,1. *pi));
end

sx::Array{ComplexF64} = [0 1 ; 1 0] 
sy::Array{ComplexF64} = [0 -1im ; 1im 0]
sz::Array{ComplexF64} = [1 0 ; 0 -1]

sm::Array{ComplexF64} = [0 0 ; 1 0]
sp::Array{ComplexF64} = [0 1 ; 0 0]

# Initial state: eigenvector of J_x with j = J
psiPSZ::Array{ComplexF64} = [0 ; 1]

sx2 = sx^2;
sy2 = sy^2;
sz2 = sz^2;

cF = sqrt(1. *Gamma1)*sm; 
cD = sqrt(1. *GammaD/2)*sz;
cPhi = sqrt(1. *GammaPhi/2)*sz;

# omega is the parameter that we want to estimate
# (we look at the case omega = 0)

# initial state of the system
RhoIn = psiPSZ * psiPSZ'

# POVM operators for strong measurement
phi=0.; #phase of the measurement
Pi1=[cos(phi)^2 sin(phi)*cos(phi) ; sin(phi)*cos(phi) sin(phi)^2];  

#
t = (1:Ntime)*dt;

rho = Array{ComplexF64}(undef, 2,2,Nomega+1);

probBayes = Array{Float64}(undef, Nomega+1, Ntime);    
lklhood = ones(Nomega + 1)/Nomega #Array{Float64}(undef, Nomega+1);

probBayesTraj = Array{Float64}(undef, Nomega+1, Ntraj,Ntime);    
lklhoodTraj = Array{Float64}(undef, Nomega+1);

omegaEst = Array{Float64}(undef, Ntime);
sigmaBayes = Array{Float64}(undef, Ntime);
omegaMaxLik = Array{Float64}(undef, Ntime);
    
AvgZcond = Array{Float64}(undef, Ntraj,Ntime)
 
for ktraj = 1:Ntraj

    dy1= real(dyHet1[:,ktraj]);
    dy2= real(dyHet2[:,ktraj]);
    dy3= real(dyDep[:,ktraj]);        
        
    for jt=1:Ntime

        M0 = I - ((cF'*cF)/2 + (cD'*cD)/2 + (cPhi'*cPhi)/2) * dt +
            sqrt(etavalF/2) * (cF * dy1[jt] - 1im * cF * dy2[jt]) + sqrt(etavalD) * (cD * dy3[jt]);
            
        for jomega = 1:(Nomega+1)
        #in questo ciclo mi calcolo le likelihood della misura al tempo jt per ciascun valore di omega
            
            H = (omegay[jomega]/2.) * sy; # Hamiltonian of the qubit
                
            if jt == 1
                rho[:,:,jomega]= RhoIn;
            end

            M = M0 - 1im * H * dt
            rhotmp = view(rho, :,:,jomega)

            newRho = M * rhotmp * M' 
            newRho += (1 - etavalF) * dt * (cF * rhotmp * cF') 
            newRho += (1 - etavalD) * dt * (cD * rhotmp * cD') 
            newRho += dt * (cPhi * rhotmp * cPhi');

            lklhood[jomega] = real(tr(newRho));
                
            rho[:,:,jomega] = newRho / lklhood[jomega];
            
            if abs(omegay[jomega]-omegaTrue) < domega    
                AvgZcond[ktraj,jt] = real(tr(rho[:,:,jomega]*sz));
            end
        end  # fine ciclo su omega
        
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
        
        
        if ktraj == Ntraj        
            omegaEst[jt] = sum(probBayes[:,jt].*omegay);
            sigmaBayes[jt] = sqrt(sum(probBayes[:,jt].*(omegay.^2)) - omegaEst[jt]^2);
                
            indM = argmax(probBayes[:,jt]);   
            omegaMaxLik[jt]=omegay[indM];
        end

    end #fine ciclo sul tempo
        
end # fine ciclo su traiettorie

return (t, omegay, AvgZcond, probBayes, probBayesTraj, omegaEst, omegaMaxLik, sigmaBayes)
end