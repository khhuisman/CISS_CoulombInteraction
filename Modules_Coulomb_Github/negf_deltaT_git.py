#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# ### Path to modules

# In[2]:


import sys
sys.path.insert(0, '/Users/khhuisman/Documents/Jupyter_notebooks/py_files')
sys.path.insert(0, '/Users/khhuisman/Documents/Jupyter_notebooks/py_files/Hubbard_Models/')



import negf



def func_deriv_zero(energies,fp_list):
    
    '''
    Input:
    energies = list of energies
    dT_list = list of the derivatives of some function f
    Returns:
    energies_zero_der = list of approximate values (x0) for which f(x0)' = 0
    '''
    
    
    energies_zero_der = [] 
    
    for i in range(1,len(energies)):
    
        dTi = fp_list[i]
        dTi1 = fp_list[i-1]

        dTi_sign = np.sign(dTi)
        dTi1_sign = np.sign(dTi1)

        if dTi_sign == 1 and dTi1_sign == -1:
            energies_zero_der.append(energies[i])



        if dTi_sign == -1 and dTi1_sign == 1:
            energies_zero_der.append(energies[i])
        
        
    return energies_zero_der




def Deltaf(energy,
          muL,muR,
          betaL,betaR):
    
    '''Returns:
    Df
    '''
    
    df = (negf.fermi_dirac(energy,muL,betaL)-negf.fermi_dirac(energy,muR,betaR))
    
    
    return df


def dE_Deltaf(energy,
          muL,muR,
          betaL,betaR):
    
    '''Returns:
    dE_Df(E)
    '''
    
    dE_Df = (negf.de_fermi_dirac(energy,muL,betaL)-negf.de_fermi_dirac(energy,muR,betaR))
    
    
    return dE_Df



def dE2_Deltaf(energy,
          muL,muR,
          betaL,betaR):
    
    '''Returns:
    dE_Df(E)
    '''
    
    dE2_Df = (negf.de2_fermi_dirac(energy,muL,betaL)-negf.de2_fermi_dirac(energy,muR,betaR))
    
    
    return dE2_Df


# In[12]:


def dE2_DeltaT(energy,
               HP,HM,
           GammaLP,GammaLM,
           GammaR):
    
    '''Returns:
    dE2_DT(E)
    '''
    
    TLRP, dETLRP,dE2TLRP = negf.dEn_Tij(energy,HP,GammaLP,GammaR )
    TLRM, dETLRM,dE2TLRM = negf.dEn_Tij(energy,HM,GammaLM,GammaR)
    
    DT = TLRP - TLRM
    dE_DT = dETLRP-dETLRM
    dE2_DT = dE2TLRP-dE2TLRM
    
    return DT,dE_DT,dE2_DT


# In[13]:


def DE2_DeltaTbar(energy,
           HP,HM,
           GammaLP,GammaLM,
           GammaR,
          muL,muR,
          betaL,betaR):
    
    '''
    Returns:
    dE2_DT(E)*Df + 2 dE_DT(E)*dE_Df + DT*dE2_Df(E)
    '''
    
    DT,dE_DT,dE2_DT = dE2_DeltaT(energy,
                                   HP,HM,
                               GammaLP,GammaLM,
                               GammaR)

        
        
    Df = Deltaf(energy,
          muL,muR,
          betaL,betaR)
    
    
    dE_Df = dE_Deltaf(energy,
          muL,muR,
          betaL,betaR)
    
    
    dE2_Df = dE2_Deltaf(energy,
          muL,muR,
          betaL,betaR)
    
    
    DTbar = DT*Df
    
    dE_Tbar = dE_DT*Df +  DT*dE_Df
    
    dE2_DTbar = dE2_DT*Df + dE_DT*dE_Df + DT*dE2_Df
    
    
    
    return DTbar,dE_Tbar,dE2_DTbar






