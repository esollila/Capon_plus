# Scenario B (Q is unknown, SOI power gamma is known) 
# ---------------------------------------------------------------
# See Section IV.B in the reference: 
#   Esa Ollila, Xavier Mestre and Elias Raninen, "Beamforming design for mini-
#       mizing the signal power estimation error", Arxiv, 2025. 
# This code reproduces Figure 4, Figure 5, and Figure 6. 
#%%

import numpy as np
import matplotlib.pyplot as plt
from utils import weight, steering_vector,create_training_data,SCM

#%%
M = 25     # nr. of sensors in the ULA
#T = 200     # snapshot size -> Figure 4
#T = 500     # snapshot size -> Figure 5
T = 100      # snapshot size -> Figure 6

DOA_src = np.array([-45.02, -30.02, -20.02,-3])
K = len(DOA_src)
A0 = steering_vector(DOA_src,M) # steering matrix
a1 = A0[:,0].reshape(-1,1) # steering vector of SOI

#%% Compute source powers 
SNR = np.arange(5,-10.5,-0.5)
gam1 = 10**(SNR/10)
gam2 = 10**((SNR-2)/10)
gam3 = 10**((SNR-4)/10)
gam4 = 10**((SNR-6)/10)
gams = np.vstack([gam1, gam2, gam3, gam4])

#%% Set-up
rng = np.random.default_rng(11111)
MC_iters = 10000
methods = ["Cap","MMSE","Cap+"]
n_methods = len(methods)
nSNR = len(SNR)
SE_NMSE = {d : np.zeros((MC_iters,nSNR)) for d in methods}
SP_MSE  = {d : np.zeros((MC_iters,nSNR)) for d in methods}
Bias    = {d : np.zeros((MC_iters,nSNR)) for d in methods}
w    = {"Cap": [], "MMSE": [],"Cap+":[]}
source_dist = "8-psk"


#%% iteration starts
for isnr in range(nSNR):
    
    gam_isnr = gams[:,isnr]
    print('{:2d} / {} , SNR= {:2.1f}'.format(isnr+1, nSNR, SNR[isnr]))
    rng = np.random.default_rng(12345)
            
    for it in range(MC_iters):

        #%% Create the primary data 
        y_soi, y_ipn, s_soi = create_training_data(T, A0, gam_isnr,rng,sources=source_dist)
        y = y_soi + y_ipn
        Sigma_hat = SCM(y) # estimate of the covariance matrix
        
        # %% Compute the adaptive Capon beamformer 
        w_base = weight(DOA_src[0], Sigma_hat) #  Sigma^-1*a 
        beta_Cap = 1 / np.real(a1.conj().T @ w_base).item()  # multiplier
        w["Cap"] = beta_Cap * w_base # adaptive Capon beamfomer 

        #%% Compute s(t) = w^H y(t) and gamma = 1/T sum | s(t) |^2
        s_base = w_base.conj().T @ y
        gam0_base = np.mean(np.abs(s_base)**2)
                
        #%%
        # MVDR: 
        gammahat_Cap = gam0_base*beta_Cap**2
        SP_MSE["Cap"][it,isnr]  = (gammahat_Cap - gam_isnr[0])**2
        Bias["Cap"][it,isnr] = (gammahat_Cap-gam_isnr[0])
        
        # In Scenario B, gamma is assumed to be known
        gamma = gam_isnr[0]

        # Compute MMSE based on known gamma and estimated gammahat_Cap
        gammahat_MMSE = gamma**2 / gammahat_Cap
        SP_MSE["MMSE"][it,isnr]  = (gammahat_MMSE - gam_isnr[0])**2
        Bias["MMSE"][it,isnr] = (gammahat_MMSE-gam_isnr[0])
        w["MMSE"] = gamma / gammahat_Cap * w["Cap"]
                
        #%%  Capon+ 
        mu4 = np.mean(np.abs(w["Cap"].conj().T @ y)**4)
        den = mu4 + gammahat_Cap**2*(T-1)
        num = T*gammahat_Cap*gamma
        gammahat_Cap_plus = num/den * gammahat_Cap
        SP_MSE["Cap+"][it,isnr]  = (gammahat_Cap_plus - gam_isnr[0])**2
        Bias["Cap+"][it,isnr] = (gammahat_Cap_plus - gam_isnr[0])
        
        beta_Cap_plus =  np.sqrt(num/den)
        w["Cap+"] = beta_Cap_plus*w["Cap"]

        #%% signal estimation NMSE
        gam0_emp = np.mean(np.abs(s_soi)**2)
        for ii,d in enumerate(w):
            s_est = w[d].conj().T @ y
            SE_NMSE[d][it,isnr] = np.mean(np.abs(s_est - s_soi)**2)/gam0_emp
                            
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
        SNR,np.mean(Bias[d],axis=0)/gams[0],
        'o-',label = methods[i], 
        linewidth=lwid, markersize=msize,
        color=colors[i]
        )
#ax[0].legend(fontsize=16,ncols=1,loc='upper right')
ax[0].set_ylabel('Relative bias, $(\hat \gamma - \gamma)/\gamma$',fontsize=17)
ax[0].grid()
ax[0].set_title(" T = {}, {:}".format(T,source_dist),fontsize=18)

for i,d in enumerate(methods):
    ax[1].plot(
        SNR,np.mean(SE_NMSE[d],axis=0),
        'o-',label = methods[i],
        linewidth=lwid, markersize=msize,
        color=colors[i]       
        )
ax[1].legend(fontsize=20,ncols=1)
ax[1].set_ylabel('Signal Estimation NMSE',fontsize=20)
ax[1].grid()

for i,d in enumerate(methods):
    ax[2].plot(
        SNR,np.mean(SP_MSE[d],axis=0)/gams[0]**2,
        'o-',label=methods[i],
        linewidth=lwid,  markersize=msize,
        color=colors[i]           
        )
#ax[2].legend(fontsize=16,ncols=4)
ax[2].set_xlabel('SNR of SOI, $\gamma/\sigma^2$ [dB]',fontsize=17), 
ax[2].set_ylabel('Power estimation NMSE',fontsize=17)
ax[2].grid()

plt.show()
