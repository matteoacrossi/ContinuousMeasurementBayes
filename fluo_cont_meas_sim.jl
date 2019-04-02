using Distributed
using SharedArrays
using LinearAlgebra
using StaticArrays

"""
This function simulates the dynamics of the superconducting qubit in Ficheux et al. for Ntraj trajectories
and returns the measured photocurrents for each measurement device and for each trajectory.

It also returns the result of a strong final measurement along z, and the expected value of sigma_z. 
"""
function parallel_fluo_continuous_measurement_het_simulation(Ntraj;
        Tfinal, # Final time
        dt, # duration of infinitesimal time
        Gamma1,   # Gamma fluoresence
        GammaD,    # Gamma dephasing controllable
        GammaPhi,  # Gamma dephasing not controllable
        etaF, #efficiency fluoresence heterodyne
        etaD, #efficiency dephasing homodyne
        phi=0, #phase of strong measurement
        omegaTrue, args...) # Real value for omega

  jtot = 1 / 2; #Total spin
  dimJ = 2; # Dimension of the corresponding Hilbert space

  #dt = Tfinal / Ntime
  Ntime = trunc(Int,Tfinal/dt);  

    sx = [0 1 ; 1. 0] .+ .0im;
    sy = [0 -1im ; 1im 0] .+ .0im;
    sz = [1 0 ; 0 -1] .+ .0im;

    sm = [0 0 ; 1 0] .+ .0im
    sp = [0 1 ; 0 0] .+ .0im

    # Initial state: eigenvector of J_x with j = J
    psiPSZ = [0 ; 1] .+ .0im;


  sx2 = sx^2;
  sy2 = sy^2;
  sz2 = sz^2;
    
  cF   = SMatrix{2,2}(sqrt(1. * Gamma1)*sm); 
  cD   = SMatrix{2,2}(sqrt(1. * GammaD/2)*sz);
  cPhi = SMatrix{2,2}(sqrt(1. * GammaPhi/2)*sz);
    
  # omega is the parameter that we want to estimate
  # (we look at the case omega = 0)

  H = SMatrix{2,2}((omegaTrue/2.) * sy); # Hamiltonian of the spin system
  dH = SMatrix{2,2}(sy/2.); # Derivative of H wrt the parameter omegaTrue

  RhoIn = SMatrix{2,2}([0.04 0 ; 0 0.96] .+ 0.0im)
    
  # POVM operators for strong measurement
  phi=0.; #phase of the measurement
  Pi1=[cos(phi)^2 sin(phi)*cos(phi) ; sin(phi)*cos(phi) sin(phi)^2];  
    
  #
  t = (1:Ntime)*dt;

  dyHet1 = SharedArray{Float64}((Ntraj,Ntime))
  dyHet2 = SharedArray{Float64}((Ntraj,Ntime))
  
  dyDep = SharedArray{Float64}((Ntraj,Ntime))
  #
  OutStrong = SharedArray{Float64}((Ntraj,Ntime))
    
  AvgZcondTrue = SharedArray{Float64}((Ntraj,Ntime))

  # Trajectory-independent part of the Kraus operator
  M0 = I - (1im * H + (cF'*cF)/2 + (cD'*cD)/2 + (cPhi'*cPhi)/2) * dt
  @sync @distributed for ktraj = 1:Ntraj
      rho = RhoIn;

      # Derivative of rho wrt the parameter
      # Initial state does not depend on the paramter

      # Vectors of Wiener increments
      WienerIncrementsF1 = sqrt(dt) * randn(1, Ntime);
      WienerIncrementsF2 = sqrt(dt) * randn(1, Ntime);
      WienerIncrementsD = sqrt(dt) * randn(1, Ntime);

      dy1 = zero(t);
      dy2 = zero(t);
      dy3 = zero(t);
      outS = zero(t);  
      #
      
      for jt=1:Ntime
          dWF1 = WienerIncrementsF1[jt];      # Wiener increment
          dWF2 = WienerIncrementsF2[jt];      # Wiener increment
          dWD = WienerIncrementsD[jt];      # Wiener increment
          

          # PRE-FEEDBACK (Conditioning)
          # Using Rouchon, Ralf Eq. (7)
          # Kraus operator
          M = M0
          
          # First three timesteps are unconditional
          if jt > 3 
              # Signal variation (Rouchon, Sec. 4.1)
              dy1[jt] = sqrt(etaF/2) * real(tr(rho*(cF+cF')))*dt + dWF1;
              dy2[jt] = sqrt(etaF/2) * real(tr(rho*(-1im*(cF-cF'))))*dt + dWF2;
              dy3[jt] = sqrt(etaD) * real(tr(rho*(cD+cD')))*dt + dWD;   
              M = M0 + sqrt(etaF/2) * (cF * dy1[jt] - 1im * cF * dy2[jt]) + sqrt(etaD) * (cD * dy3[jt]);
            end
            
          newRho = M * rho * M' + (1 - etaF) * dt * (cF * rho * cF') + (1 - etaD) * dt * (cD * rho * cD') + dt * (cPhi * rho * cPhi');
          tr1 = tr(newRho);

          rho = newRho / tr1;
            
          AvgZcondTrue[ktraj,jt] = real(tr(rho*sz));
            
          # outputstrong measurement
          p1=real(tr(rho*Pi1)); 
          
          rr=rand();
          if rr < p1
               outS[jt]=1;
          else
               outS[jt]=-1;
          end
          
      end
        
      dyHet1[ktraj,:] = dy1;
      dyHet2[ktraj,:] = dy2;
      dyDep[ktraj,:] = dy3;
      OutStrong[ktraj,:]=outS;  
  end

  # Collect is required, otherwise we are 
  # returning SharedArrays; instead we want to return 
  # standard arrays
  return (t=t, 
      Ntime=Ntime, 
      dyHet1=collect(dyHet1'), 
      dyHet2=collect(dyHet2'), 
      dyDep=collect(dyDep'), 
      OutStrong=collect(OutStrong'), 
      AvgZcondTrue=collect(AvgZcondTrue))
end