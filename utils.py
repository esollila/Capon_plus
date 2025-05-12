#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils functions for the paper Ollila, Mestre and Raninen (2025)
"""

import numpy as np
from scipy.linalg import solve


def weight(theta,R):
     sv = steering_vector(theta,R.shape[0]) 
     return solve(R,sv ,assume_a='pos') 
 
def steering_vector(angle_deg,m_elems):        
    return np.exp(-1j*np.pi*np.arange(m_elems).reshape(-1,1) * np.sin(np.deg2rad(angle_deg)))

def create_training_data(T,A0,gam_isnr,rng,sources="gaussian"):

    M,K = np.shape(A0)
    noise = (rng.standard_normal((M,T))+1j*rng.standard_normal((M,T)))/np.sqrt(2)
    if sources == "gaussian":
        s = np.diag(np.sqrt(gam_isnr)) @ (rng.standard_normal((K,T))+1j*rng.standard_normal((K,T)))/np.sqrt(2)
    elif sources == "8-psk":        
        PSKsym = 8  # 8-PSK source signals
        dataIn = rng.integers(0, PSKsym, size=(K, T))
        theta = 2 * np.pi * dataIn / PSKsym
        ini_phase = np.pi / 7   
        s = np.diag(np.sqrt(gam_isnr)) @ np.exp(1j * (theta + ini_phase))
    else: 
        raise ValueError(f"Sources argument {sources} not recognized")

    y_soi = A0[:,0].reshape(-1,1) * s[0].reshape(1, -1)
    y_ipn = A0[:, 1:] @ s[1:, :] + noise
    return y_soi, y_ipn, s[0]

def SCM(X):
    _,T = np.shape(X)
    est  = (1/T)*X @ X.conj().T # SCM
    return est