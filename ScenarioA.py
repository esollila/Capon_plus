# Scenario A (INCM Q is known, gamma is unknown) 
# ----------------------------------------------
# See Section IV.A of Ollila, Mestre and Raninen (2025)
# This code reproduces Figure corresponding the Scenario A in the paper. 
#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from utils import weight, steering_vector, create_training_data

#%%
M = 25     # nr. of sensors in the ULA
T = 60     # snapshot size

#%% Compute source powers 
SNR = np.arange(0,-10.5,-0.5)
gam1 = 10**(SNR/10)
gam2 = 10**((SNR-2)/10)
gam3 = 10**((SNR-4)/10)
gam4 = 10**((SNR-6)/10)
gams = np.vstack([gam1, gam2, gam3, gam4])

#%% Set-up
rng = np.random.default_rng(11111)
MC_iters = 20000
methods = ["Cap","MMSE","Cap+"]
n_methods = len(methods)
nSNR = len(SNR)
SE_NMSE  = {d : np.zeros((MC_iters,nSNR)) for d in methods}
SP_MSE  = {d : np.zeros((MC_iters,nSNR)) for d in methods}
Bias    = {d : np.zeros((MC_iters,nSNR)) for d in methods}
w    = {"Cap": [], "MMSE": [],"Cap+":[]}
source_dist="8-psk"
gammahat_deb = np.zeros((MC_iters,nSNR))
SP_MSE["deb"] = np.zeros((MC_iters,nSNR))

#%% iteration starts

for isnr in range(nSNR):
    gam_isnr = gams[:, isnr]
    # rng = np.random.default_rng(12345)
    print('{:2d} / {} , SNR= {:4.1f}'.format(isnr + 1, nSNR, SNR[isnr]))

    for it in range(MC_iters):

        # SOI DOA
        DOA_src = np.array([-45])

        # Interferences uniformly random in [-90, DOA-phi] V [DOA+phi, 90]
        phi = 3
        interval = np.array([-90,90-2*phi])
        random_angles = np.random.uniform(low=interval[0], high=interval[1], size=[3,])
        for val in random_angles:
            if val < DOA_src[0]-phi:
                interf_angle = val
                # append interferences to DOA_src
                DOA_src = np.concatenate((DOA_src, np.array([interf_angle])), axis=0)
            else:
                interf_angle = val + 2*phi
                # append interferences to DOA_src
                DOA_src = np.concatenate((DOA_src, np.array([interf_angle])), axis=0)

        K = len(DOA_src)
        A0 = steering_vector(DOA_src, M)  # steering matrix

        # Compute Covariance matrix and INCM Q
        Cov = A0 @ np.diag(gam_isnr) @ A0.conj().T + np.eye(M)
        iCov = solve(Cov,np.eye(M) ,assume_a='pos')
        Q = A0[:,1:] @ np.diag(gam_isnr[1:]) @ A0[:,1:].conj().T + np.eye(M)
        iQ  = solve(Q,np.eye(M) ,assume_a='pos')

        # Steering vector for SOI is known
        DOA_SOI = DOA_src[0]
        a1 = steering_vector(DOA_SOI, M)

        # Compute weight of Capon
        w_base = weight(DOA_SOI,Q)
        beta_Cap =  1/np.real(a1.conj().T @ w_base).item() # multiplier
        w["Cap"] = beta_Cap*w_base # w cap is known and does not depend on data
        
        # Create the data
        y_soi, y_ipn, s_soi = create_training_data(T, A0, gam_isnr,rng,sources=source_dist)
        y = y_soi + y_ipn    
                
        #    
        s_base = w_base.conj().T @ y
        gam0_base = np.mean(np.abs(s_base)**2)
                
        # MVDR: 
        gammahat_Cap = gam0_base*beta_Cap**2
        SP_MSE["Cap"][it,isnr]  = (gammahat_Cap - gam_isnr[0])**2
        Bias["Cap"][it,isnr] = (gammahat_Cap-gam_isnr[0])
        
        # Debiased estimator that we will use as an estimator of gamma
        gammahat = gammahat_Cap - beta_Cap
        gammahat = np.max([gammahat,0])

        gammahat_deb[it,isnr] = gammahat
        SP_MSE["deb"][it,isnr]  =  (gammahat - gam_isnr[0])**2
        
        # MMSE (using the estimated gammahat): 
        beta_MMSE = gammahat/(1+gammahat/beta_Cap)
        gammahat_MMSE =  beta_MMSE**2 * gam0_base
        SP_MSE["MMSE"][it,isnr]  = (gammahat_MMSE - gam_isnr[0])**2
        Bias["MMSE"][it,isnr] = (gammahat_MMSE-gam_isnr[0])
        w["MMSE"] = beta_MMSE*w_base
                
        #  Capon+ for Scenario A (INCM Q known, gamma_SOI is unknown)
        mu4 = np.mean(np.abs(w["Cap"].conj().T @ y)**4)
        den = mu4 + gammahat_Cap**2*(T-1)
        num = T*gammahat_Cap*gammahat
        gammahat_Cap_plus = num/den * gammahat_Cap
        SP_MSE["Cap+"][it,isnr]  = (gammahat_Cap_plus - gam_isnr[0])**2
        Bias["Cap+"][it,isnr] = (gammahat_Cap_plus - gam_isnr[0])
        
        beta_Cap_plus =  np.sqrt(num/den)
        w["Cap+"] = beta_Cap_plus*w["Cap"]

        # signal estimation NMSE
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
        SNR, np.mean(Bias[d],axis=0)/gams[0],
        'o-', label = methods[i], 
        linewidth=lwid, markersize=msize,
        color=colors[i]
        )
ax[0].plot(
    SNR, np.mean(gammahat_deb-gams[0],axis=0)/gams[0],
    'x', label = "deb", 
    linewidth=lwid, markersize=msize,
    )
#ax[0].legend(fontsize=20,ncols=6,loc='upper right')
ax[0].set_ylabel('Relative bias, $(\hat \gamma - \gamma)/\gamma$',fontsize=18)
ax[0].set_title("T = {}, {}".format(T,source_dist),fontsize=18)
ax[0].grid()

for i,d in enumerate(methods):
    ax[1].plot(
        SNR,np.mean(SE_NMSE[d],axis=0),
        'o-',label = methods[i],
        linewidth=lwid, markersize=msize,
        color=colors[i]
        )
#ax[1].legend(fontsize=20,ncols=1)
ax[1].set_ylabel('Signal Estimation NMSE',fontsize=18)
ax[1].grid()

for i,d in enumerate(methods):
    ax[2].plot(
        SNR,np.mean(SP_MSE[d],axis=0)/gams[0]**2,
        'o-', label=methods[i],
        linewidth=lwid, markersize=msize, 
        color=colors[i]
        )  
ax[2].plot(
    SNR, np.mean(SP_MSE["deb"],axis=0)/gams[0]**2,
    'x-', label = "deb", linewidth=lwid, markersize=msize
    )
ax[2].legend(fontsize=16,ncols=1)
ax[2].set_xlabel('SNR of SOI, $\gamma/\sigma^2$ [dB]',fontsize=17)
ax[2].set_ylabel('Signal power NMSE',fontsize=18)
ax[2].grid()
plt.show()


# %%
