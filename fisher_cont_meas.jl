using ZChop # For chopping small imaginary parts in ρ
using SharedArrays

"""
    QFI(ρ, dρ [, abstol])
Numerically evaluate the quantum Fisher information for the matrix ρ given its derivative dρ wrt the parameter
This function is the implementation of Eq. (13) in Paris, Int. J. Quantum Inform. 7, 125 (2009).
# Arguments
    * `ρ`:  Density matrix
    * `dhro`: Derivative wrt the parameter to be estimated
    * `abstol = 1e-5`: tolerance in the denominator of the formula
"""
function QFI(ρ, dρ; abstol = 1e-5)
    # Get the eigenvalues and eigenvectors of the density matrix
    # We enforce its Hermiticity so that the algorithm is more efficient and returns real values
    eigval, eigvec = eigen(Matrix(zchop.(ρ,1e-10)))
    eigval = real(eigval)
    dim = length(eigval)
    return real(2*sum( [( (eigval[n] + eigval[m] > abstol) ? (1. / (eigval[n] + eigval[m])) * abs(eigvec[:,n]' * dρ * eigvec[:,m])^2 : 0.) for n=1:dim, m=1:dim]))
end

function parallel_fluo_continuous_measurement_het_classic_initial0(Ntraj;
        # i will use all the data of the experiment taking as a unit of time 1 microsecond               
        Tfinal = nothing, # Final time
        Ntime = nothing, # Number of timesteps
        Gamma1 = nothing,   # Gamma fluoresence
        GammaD = nothing,    # Gamma dephasing controllable
        GammaPhi = nothing,  # Gamma dephasing not controllable
        etaF = nothing, #efficiency fluoresence heterodyne
        etaD = nothing, #efficiency dephasing homodyne
        phi = nothing, #phase of the strong measurement    
        omegaTrue = nothing, params...) # omegaz

    omegay = omegaTrue
  jtot = 1 / 2; #Total spin
  dimJ = 2; # Dimension of the corresponding Hilbert space

  dt = Tfinal / Ntime

  sx = [0 1 ; 1 0];
  sy = [0 -1im ; 1im 0];
  sz = [1 0 ; 0 -1];

  sm = [0 0 ; 1 0];
  sp = [0 1 ; 0 0];
    
  psiPSZ = [0 ; 1] ;

  sx2 = sx^2;
  sy2 = sy^2;
  sz2 = sz^2;
    
  cF = SMatrix{2,2}(sqrt(1. *Gamma1)*sm)
  cD = SMatrix{2,2}(sqrt(1. *GammaD/2)*sz)
  cPhi = SMatrix{2,2}(sqrt(1. *GammaPhi/2)*sz)
    
  # omega is the parameter that we want to estimate
  # (we look at the case omega = 0)

  H = SMatrix{2,2}((omegay/2.) * sy) # Hamiltonian of the spin system
  dH = SMatrix{2,2}(sy/2.) # Derivative of H wrt the parameter omegax

  # Initial state
  RhoIn = SMatrix{2,2}([0.04 0 ; 0 0.96] .+ 0.0im);
  
  # POVM operators for strong measurement
  phi=0.; #phase of the measurement
  Pi1=[cos(phi)^2 sin(phi)*cos(phi) ; sin(phi)*cos(phi) sin(phi)^2];
    
  #
  t = (1:Ntime)*dt;

  FisherTraj = Array{Float64}(undef, Ntraj,Ntime)
  #  
  QFisherTraj = Array{Float64}(undef, Ntraj,Ntime)
  #
  FisherMTraj = Array{Float64}(undef, Ntraj,Ntime)
  #
    
  FisherAvg = Array{Float64}(undef, Ntime)
  #  
  QFisherAvg = Array{Float64}(undef, Ntime)
  #  
  QFisherEff = Array{Float64}(undef, Ntime)
  #
  FisherMAvg = Array{Float64}(undef, Ntime)
  FisherMEff = Array{Float64}(undef, Ntime)
    
  for ktraj = 1:Ntraj
      rho = RhoIn

      # Derivative of rho wrt the parameter
      # Initial state does not depend on the paramter
      Drho = zero(rho)
      tau = zero(Drho)

      # Vectors of Wiener increments
      WienerIncrementsF1 = sqrt(dt) * randn(1, Ntime);
      WienerIncrementsF2 = sqrt(dt) * randn(1, Ntime);

      WienerIncrementsD = sqrt(dt) * randn(1, Ntime);

      Fisher = zeros(length(t));
      QFisher = zeros(length(t));
      FisherM = zeros(length(t));
      #
        
      for jt=1:Ntime
          dWF1 = WienerIncrementsF1[jt];      # Wiener increment
          dWF2 = WienerIncrementsF2[jt];      # Wiener increment

          dWD = WienerIncrementsD[jt];      # Wiener increment
            
          # We initialize dy1 and dy2 
          # Signal variation (Rouchon, Sec. 4.1)
          dy1 = sqrt(etaF/2) * tr(rho*(cF+cF'))*dt + dWF1;
          dy2 = sqrt(etaF/2) * tr(rho*(-1im*(cF-cF')))*dt + dWF2;
            
          dy3 = sqrt(etaD) * tr(rho*(cD+cD'))*dt + dWD;   

          #  PRE-FEEDBACK (Conditioning)
          # Using Rouchon, Ralf Eq. (7)
          # Kraus operator
          M = I - (1im * H + (cF'*cF)/2 + (cD'*cD)/2 + (cPhi'*cPhi)/2) * dt +
              sqrt(etaF/2) * (cF * dy1 - 1im * cF * dy2) + sqrt(etaD) * (cD * dy3);
          # Derivative of the Kraus operator
          dM = -1im * dH * dt;
            
          newRho = M * rho * M' + (1 - etaF) * dt * (cF * rho * cF') + (1 - etaD) * dt * (cD * rho * cD') + dt * (cPhi * rho * cPhi');
          tr1 = tr(newRho);
          tau = (M * tau * M' + dM * rho * M' + M * rho * dM' + (1 - etaF) * dt * (cF * tau * cF') +  (1 - etaD) * dt * (cD * tau * cD') + dt * (cPhi * tau * cPhi')) / tr1;

          rho = newRho / tr1;
          Drho = tau - tr(tau) * rho;
          
            
          # We evaluate the quantities of interest
          Fisher[jt] = real(tr(tau))^2;
          #  
          QFisher[jt] = QFI(rho,Drho);
          #
          # classical Fisher of strong measurement
          p1=real(tr(rho*Pi1));
          p2=1-p1;
          #  
          #    
          FisherM[jt] = real((real(tr(Drho*Pi1))^2)*(1/p1+1/p2)); 
          #  

      end
        
      FisherTraj[ktraj,:] = Fisher;
      QFisherTraj[ktraj,:] = QFisher;
      FisherMTraj[ktraj,:] = FisherM;
  end

  for jt = 1:Ntime  
      FisherAvg[jt] = sum(FisherTraj[:,jt])/Ntraj;
      QFisherAvg[jt] = sum(QFisherTraj[:,jt])/Ntraj;
      FisherMAvg[jt] = sum(FisherMTraj[:,jt])/Ntraj; 
  end
  #   

  QFisherEff = FisherAvg + QFisherAvg;
  FisherMEff = FisherAvg + FisherMAvg;
  #  
  return (t=t, FisherAvg=FisherAvg, QFisherAvg=QFisherAvg, FisherMAvg=FisherMAvg', QFisherEff=QFisherEff,FisherMEff=FisherMEff)
end