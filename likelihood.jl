using LinearAlgebra

PriorGaussian(omega, omegaMean, Sigma) = exp.( - ((omega .- omegaMean).^2)/(2*Sigma^2))

# in questa funzione faccio stima bayesiana sulle singole traiettorie 
# non moltiplico probablità tra traiettorie diverse
# non considero misure proiettiva finale (sembrerebbe inutile)

function Likelihoood(Ntraj, Ntime;
    Tfinal = 10., # Final time
    Gamma1 = 1. / 15 ,   # Gamma fluoresence
    GammaD = 1. / (0.3),    # Gamma dephasing controllable
    GammaPhi = 1. / (17.9),  # Gamma dephasing not controllable
    etavalF=0.14, #efficiency fluoresence heterodyne
    etavalD=0.34, #efficiency dephasing homodyne
    omegaMin=-1. * pi, #minimum value of omega
    omegaMax=1. * pi, #maximum value of omega
    Nomega = 500, # number of values of omega
    omegaTrue = 2. * pi / 5.,  #true value of omega
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
priorG[jomega] = real(PriorGaussian(omegay[jomega],omegaMax/2.,omegaMax/1.));
end

sx = [0 1 ; 1 0];
sy = [0 -1im ; 1im 0];
sz = [1 0 ; 0 -1];

sm = [0 0 ; 1 0];
sp = [0 1 ; 0 0];

# Initial state: eigenvector of J_x with j = J
psiPSZ = [0 ; 1] ;

sx2 = sx^2;
sy2 = sy^2;
sz2 = sz^2;

cF = sqrt(1. * Gamma1) * sm; 
cD = sqrt(1. * GammaD/2) * sz;
cPhi = sqrt(1. * GammaPhi/2) * sz;

# omega is the parameter that we want to estimate
# (we look at the case omega = 0)

# initial state of the system
RhoIn = psiPSZ * psiPSZ'

# POVM operators for strong measurement
phi=0.; #phase of the measurement
Pi1=[cos(phi)^2 sin(phi)*cos(phi) ; sin(phi)*cos(phi) sin(phi)^2];  

#
t = (1:Ntime)*dt;
    

rho = Array{ComplexF64}(undef,2,2,Nomega+1);
newRho = Array{ComplexF64}(undef, 2,2,Nomega+1);

probBayes = Array{Float64}(undef, Nomega+1,Ntraj,Ntime);    
lklhood = Array{Float64}(undef, Nomega+1);

omegaEst = Array{Float64}(undef, Ntraj,Ntime);
sigmaBayes = Array{Float64}(undef, Ntraj,Ntime);
omegaMaxLik = Array{Float64}(undef, Ntraj,Ntime);
    
AvgZcond = Array{Float64}(undef, Ntraj,Ntime)

#rho = zeros(2,2,Nomega+1);
#newRho = zeros(2,2,Nomega+1);

#probBayes = zeros(Nomega+1,Ntraj,Ntime);    
#lklhood = zeros(Nomega+1);

#omegaEst = zeros(Ntraj,Ntime);
#sigmaBayes = zeros(Ntraj,Ntime);
 
for ktraj = 1:Ntraj

    dy1= real(dyHet1[:,ktraj]);
    dy2= real(dyHet2[:,ktraj]);
    dy3= real(dyDep[:,ktraj]);        
        
    for jt=1:Ntime
                        
        for jomega = 1:(Nomega+1)
        #in questo ciclo mi calcolo le likelihood della misura al tempo jt per ciascun valore di omega
            
            H = (omegay[jomega]/2.) * sy; # Hamiltonian of the qubit
                
            if jt == 1
                rho[:,:,jomega]= RhoIn;
            end

            M = I - (1im * H + (cF'*cF)/2 + (cD'*cD)/2 + (cPhi'*cPhi)/2) * dt +
                sqrt(etavalF/2) * (cF * dy1[jt] - 1im * cF * dy2[jt]) + sqrt(etavalD) * (cD * dy3[jt]);
            
            newRho[:,:,jomega] = M * rho[:,:,jomega] * M' + (1 - etavalF) * dt * (cF * rho[:,:,jomega] * cF') + (1 - etavalD) * dt * (cD * rho[:,:,jomega] * cD') + dt * (cPhi * rho[:,:,jomega] * cPhi');
            
            lklhood[jomega] = real(tr(newRho[:,:,jomega]));
                
            rho[:,:,jomega] = newRho[:,:,jomega]/lklhood[jomega];
            
            if abs(omegay[jomega]-omegaTrue) < domega    
                AvgZcond[ktraj,jt] = real(tr(rho[:,:,jomega]*sz));
                # qui mi salvo dei valori medi per avere un controllo che tutto funzioni con la simulazione dell'esperimento sopra
                # e sembra funzionare tutto
            end
        end  # fine ciclo su omega
        
        if jt == 1
            lklhood[:] = lklhood[:]#.*priorG[:];
        else
            lklhood[:] = lklhood[:].*probBayes[:,ktraj,jt-1];
        end
        
        norm = sum(lklhood[:]);

        probBayes[:,ktraj,jt]=lklhood[:]/norm;
        
        # qui sopra mi accorgo che ciclo tante volte su jomega, ma non ho trovato un metodo semplice migliore
        
        # se dovessi usare tutte le traiettorie per stimare, 
        # quello che faccio qui sotto non va bene...ma per adesso lasciamo così, 
        # la mia speranza è che serva una sola traiettoria
            
        omegaEst[ktraj,jt] = sum(probBayes[:,ktraj,jt].*omegay);
        sigmaBayes[ktraj,jt] = sqrt(sum(probBayes[:,ktraj,jt].*(omegay.^2)) - omegaEst[ktraj,jt]^2);

        tmp, indM = findmax(probBayes[:,ktraj,jt]);   
        omegaMaxLik[ktraj,jt]=omegay[indM];
        # mi calcolo anche l'omega corrispondente all'omegaMaxLik    

    end #fine ciclo sul tempo
        
end # fine ciclo su traiettorie

return (t, omegay, AvgZcond, probBayes, omegaEst, omegaMaxLik, sigmaBayes)
end