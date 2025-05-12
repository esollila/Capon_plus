# Scenario C (Q is unknown, gamma is known, secondary data available) 
# -------------------------------------------------------------------
# See Section IV.C in the reference: 
#   Esa Ollila, Xavier Mestre and Elias Raninen, "Beamforming design for mini-
#       mizing the signal power estimation error", Arxiv, 2025. 
# This code reproduces 
# Figure 7: Sample size T_0 for INCM estimation vs 
#           Relative Bias/Signal estimation NMSE/Power estimation MSE
#%%
import numpy as np
import matplotlib.pyplot as plt
from utils import weight,steering_vector,create_training_data

#%%
M = 25     # nr. of sensors in the ULA
# snapshot size T_0 for estimating INCM Q varies from 30 to 120
T0arr = np.concatenate([np.arange(30, 55, 5), np.arange(60, 130, 10)]) 
T = 60    # snapshot size (data with SOI)
DOA_src = np.array([-45.02, -30.02, -20.02,-3])
K = len(DOA_src)
A0 = steering_vector(DOA_src,M) # steering matrix
a1 = A0[:,0].reshape(-1,1) # steering vector of SOI

#%% Compute source powers 
SNR = -5
gam1 = 10**(SNR/10)
gam2 = 10**((SNR-2)/10)
gam3 = 10**((SNR-4)/10)
gam4 = 10**((SNR-6)/10)
gams = np.array([gam1, gam2, gam3, gam4])

#%% Set-up
rng = np.random.default_rng(11111)
MC_iters = 10000
methods = ["Cap","MMSE","Cap+"]
n_methods = len(methods)
nT0 = len(T0arr)
SE_NMSE = {d : np.zeros((MC_iters,nT0)) for d in methods}
SP_MSE  = {d : np.zeros((MC_iters,nT0)) for d in methods}
Bias    = {d : np.zeros((MC_iters,nT0)) for d in methods}
w    = {"Cap": [], "MMSE": [],"Cap+":[]}
source_dist = "8-psk"

#%% iteration starts
for iT0 in range(nT0):

    # sample size for weight vector estimation
    T0 = T0arr[iT0]

    print('{:2d} / {} , nSamples= {:d}'.format(iT0 + 1, nT0, T0arr[iT0]))
    rng = np.random.default_rng(12345)
        
    for it in range(MC_iters):

        #%% Create the  data with T snapshots
        y_soi, y_ipn, s_soi = create_training_data(T, A0, gams,rng,sources=source_dist)
        y = y_soi + y_ipn

        #%% Create the secondary data with T_0 snapshots for estimating INCM Q 
        z_soi, z_ipn, _ = create_training_data(T0, A0, gams,rng,sources=source_dist)
        y0 = z_ipn # secondary data without the SOI
        Qhat = y0 @ np.conj(y0).T / T0

        # %% Compute weight of Capon using the eestimated Q 
        w_base = weight(DOA_src[0], Qhat)      # Q^-1*a
        beta_Cap = 1 / np.real(a1.conj().T @ w_base).item() # 1/(a^H*Q^-1*a)
        w["Cap"] = beta_Cap * w_base # Capon's weight

        #%%
        s_base = w_base.conj().T @ y
        gam0_base = np.mean(np.abs(s_base)**2)
                
        #%%
        # MVDR: 
        gammahat_Cap = gam0_base*beta_Cap**2
        SP_MSE["Cap"][it,iT0]  = (gammahat_Cap - gams[0]) ** 2
        Bias["Cap"][it,iT0] = (gammahat_Cap - gams[0])
        
        # In Scenario C, we assume known SOI signal power 
        gamma = gams[0] # power of the SOI at DOA_src[0]
        
        # Compute MMSE beamformer based on known gamma and estimated gammahat_Cap
        gammahat_MMSE = gamma**2 / gammahat_Cap
        SP_MSE["MMSE"][it,iT0]  = (gammahat_MMSE - gams[0]) ** 2
        Bias["MMSE"][it,iT0] = (gammahat_MMSE - gams[0])
        w["MMSE"] = gamma / gammahat_Cap * w["Cap"]
                
        #%%  Capon+ beamformer 
        mu4 = np.mean(np.abs(w["Cap"].conj().T @ y)**4)
        den = mu4 + gammahat_Cap**2*(T-1)
        num = T*gammahat_Cap*gamma
        gammahat_Cap_plus = num/den * gammahat_Cap
        SP_MSE["Cap+"][it,iT0]  = (gammahat_Cap_plus - gams[0]) ** 2
        Bias["Cap+"][it,iT0] = (gammahat_Cap_plus - gams[0])
        
        beta_Cap_plus =  np.sqrt(num/den)
        w["Cap+"] = beta_Cap_plus*w["Cap"]

        #%% signal estimation NMSE
        gam0_emp = np.mean(np.abs(s_soi)**2)
        for ii,d in enumerate(w):
            s_est = w[d].conj().T @ y
            SE_NMSE[d][it,iT0] = np.mean(np.abs(s_est - s_soi) ** 2) / gam0_emp
                            
#%%  Plotting
lwid = 1.0
msize = 8
colors = [
    np.array([31, 119, 180]) / 255,   # Capon
    np.array([255, 127, 14]) / 255,   # MMSE
    np.array([214, 39, 40]) / 255     # Capon+
]
fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(10,18))

for i,d in enumerate(methods):
    ax[0].plot(
        T0arr,np.mean(Bias[d],axis=0)/gams[0],
        'o-',label = methods[i], 
        linewidth=lwid, markersize=msize,
        color=colors[i]
        )
#ax[0].legend(fontsize=16,ncols=1,loc='upper right')
ax[0].set_ylabel('Relative bias, $(\hat \gamma - \gamma)/\gamma$',fontsize=17)
ax[0].set_title("SNR (soi)= {}, T = {}, {}".format(SNR,T,source_dist),fontsize=18)
ax[0].grid()

for i,d in enumerate(methods):
    ax[1].plot(
        T0arr,np.mean(SE_NMSE[d],axis=0),
        'o-',label = methods[i],
        linewidth=lwid, markersize=msize,
        color=colors[i]
        )
ax[1].legend(fontsize=18,ncols=1)
ax[1].set_ylabel('Signal Estimation NMSE',fontsize=17)
ax[1].grid()


for i,d in enumerate(methods):
    ax[2].plot(
        T0arr,np.mean(SP_MSE[d],axis=0)/gams[0]**2,
        'o-',label=methods[i],
        linewidth=lwid, markersize=msize,
        color=colors[i]
        )
#ax[2].legend(fontsize=16,ncols=1)
ax[2].set_xlabel('Sample size $T_0$ for INCM estimation',fontsize=17),
ax[2].set_ylabel('Power estimation NMSE',fontsize=17)
ax[2].grid()
plt.show()
