#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt




# ### Path to modules

# In[2]:



import negf_HIA_git as negf_HIA




# # Estimation of error with trapezoid method



def DeltaT(energy,Hamiltonian0,
           GammaLP,GammaLM,
           GammaR,U,
           nPlist,nMlist):
    
    '''Returns:
    dT(E)
    '''
    
    TP = negf_HIA.TLR(energy,Hamiltonian0,GammaLP,GammaR,U,nPlist )
    TM = negf_HIA.TLR(energy,Hamiltonian0,GammaLM,GammaR,U,nMlist )
    
    return TP-TM



def Deltaf(energy,
          muL,muR,
          betaL,betaR):
    
    '''Returns:
    Df
    '''
    
    df = (negf_HIA.fermi_dirac(energy,muL,betaL)-negf_HIA.fermi_dirac(energy,muR,betaR))
    
    
    return df


def dT_prime(energy,Hamiltonian0,
           GammaLP,GammaLM,
           GammaR,U,
           nPlist,nMlist):
    
    '''Returns:
    dE_dT(E)
    '''
    
    dETLRP = negf_HIA.dETLR(energy,Hamiltonian0,GammaLP,GammaR,U,nPlist )
    dETLRM = negf_HIA.dETLR(energy,Hamiltonian0,GammaLM,GammaR,U,nMlist )
    dTprime = dETLRP-dETLRM
    
    return dTprime








def dE_Deltaf(energy,
          muL,muR,
          betaL,betaR):
    
    '''Returns:
    dE_Df(E)
    '''
    
    dE_Df = (negf_HIA.de_fermi_dirac(energy,muL,betaL)-negf_HIA.de_fermi_dirac(energy,muR,betaR))
    
    
    return dE_Df



def dE2_Deltaf(energy,
          muL,muR,
          betaL,betaR):
    
    '''Returns:
    dE_Df(E)
    '''
    
    dE2_Df = (negf_HIA.de2_fermi_dirac(energy,muL,betaL)-negf_HIA.de2_fermi_dirac(energy,muR,betaR))
    
    
    return dE2_Df



def DE_Tbar(energy,
           Hamiltonian0,
           GammaLP,GammaLM,
           GammaR,U,
           nPlist,nMlist,
          muL,muR,
          betaL,betaR):
    
    '''Returns:
    dE_DT(E)*Df + DT*dE_Df(E)
    '''
    
    gprime = dT_prime(energy,Hamiltonian0,GammaLP,GammaLM,GammaR,U,nPlist,nMlist)*Deltaf(energy, muL,muR,betaL,betaR)              + DeltaT(muL,Hamiltonian0, GammaLP,GammaLM,GammaR,U,nPlist,nMlist)*Deltaf_prime(energy, 
                                                                                                  muL,muR,
                                                                                                  betaL,betaR)
    
    
    
    return gprime



# In[20]:


def dE2_DeltaT(energy,Hamiltonian0,
           GammaLP,GammaLM,
           GammaR,U,
           nPlist,nMlist):
    
    '''Returns:
    dE2_DT(E)
    '''
    
    TLRP, dETLRP,dE2TLRP = negf_HIA.dEn_TLR(energy,Hamiltonian0,GammaLP,GammaR,U,nPlist )
    TLRM, dETLRM,dE2TLRM = negf_HIA.dEn_TLR(energy,Hamiltonian0,GammaLM,GammaR,U,nMlist )
    
    DT = TLRP - TLRM
    dE_DT = dETLRP-dETLRM
    dE2_DT = dE2TLRP-dE2TLRM
    
    return DT,dE_DT,dE2_DT
    


# In[21]:


def DE2_DeltaTbar(energy,
           Hamiltonian0,
           GammaLP,GammaLM,
           GammaR,U,
           nPlist,nMlist,
          muL,muR,
          betaL,betaR):
    
    '''
    Returns:
    dE2_DT(E)*Df + 2 dE_DT(E)*dE_Df + DT*dE2_Df(E)
    '''
    
    DT,dE_DT,dE2_DT = dE2_DeltaT(energy,Hamiltonian0,
                                           GammaLP,GammaLM,
                                           GammaR,U,
                                           nPlist,nMlist)
        
        
        
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


def DE2_DeltaTbar_error_formula(energy,
           Hamiltonian0,
           GammaLP,GammaLM,
           GammaR,U,
           nPlist,nMlist,
          muL,muR,
          betaL,betaR):
    
    '''
    Returns:
    dE2_DT(E)*Df + 2 dE_DT(E)*dE_Df + DT*dE2_Df(E)
    '''
    
    DT,dE_DT,dE2_DT = dE2_DeltaT(energy,Hamiltonian0,
                                           GammaLP,GammaLM,
                                           GammaR,U,
                                           nPlist,nMlist)
        
        
        
    Df = Deltaf(energy,
          muL,muR,
          betaL,betaR)
    
    
    dE_Df = dE_Deltaf(energy,
          muL,muR,
          betaL,betaR)
    
    
    dE2_Df = dE2_Deltaf(energy,
          muL,muR,
          betaL,betaR)
    
    
    
    dE2_DTbar = dE2_DT*Df + dE_DT*dE_Df + DT*dE2_Df
    
    
    
    return dE2_DTbar




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





def energies_max_deltaT(energies,Hamiltonian0,
               GammaLP,GammaLM,
               GammaR,U,
               nPlist,nMlist,muL,muR,betaL,betaR):
    
    
    '''
    Output:
    - List of energies where derivative of DTbar is zero.
    '''
    
    DeltaTbar = []
    dTbar_list = []
    d2Tbar_list = []
    
    for i in range(len(energies)):

        energy = energies[i]

        DTbar,dE_Tbar,dE2_DTbar = DE2_DeltaTbar(energy,Hamiltonian0,
               GammaLP,GammaLM,
               GammaR,U,
               nPlist,nMlist,muL,muR,betaL,betaR)


        DeltaTbar.append(DTbar)

        dTbar_list.append(dE_Tbar)
        d2Tbar_list.append(dE2_DTbar)






    energies_zero_der = func_deriv_zero(energies,dTbar_list)
    energies_zero_der2 = func_deriv_zero(energies,d2Tbar_list)
        
    return energies_zero_der,energies_zero_der2



def dE2_barT(energy,Hamiltonian0,
           GammaLP,GammaLM,
           GammaR,U,
           nPlist,nMlist):
    
    '''Returns:
    dE2_DT(E)
    '''
    
    TLRP, dETLRP,dE2TLRP = negf_HIA.dEn_TLR(energy,Hamiltonian0,GammaLP,GammaR,U,nPlist )
    TLRM, dETLRM,dE2TLRM = negf_HIA.dEn_TLR(energy,Hamiltonian0,GammaLM,GammaR,U,nMlist )
    
    barT = TLRP + TLRM
    dE_barT = dETLRP + dETLRM
    dE2_barT = dE2TLRP + dE2TLRM
    
    return barT,dE_barT,dE2_barT


def DE2_BarTbar(energy,
           Hamiltonian0,
           GammaLP,GammaLM,
           GammaR,U,
           nPlist,nMlist,
          muL,muR,
          betaL,betaR):
    
    '''
    Returns:
    dE2_DT(E)*Df + 2 dE_DT(E)*dE_Df + DT*dE2_Df(E)
    '''
    
    DT,dE_DT,dE2_DT = dE2_barT(energy,Hamiltonian0,
                                           GammaLP,GammaLM,
                                           GammaR,U,
                                           nPlist,nMlist)
        
        
        
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



def energies_max_barT(energies,Hamiltonian0,
               GammaLP,GammaLM,
               GammaR,U,
               nPlist,nMlist,muL,muR,betaL,betaR):
    
    
    '''
    Output:
    - List of energies where derivative of DTbar is zero.
    '''
    
    BTbar_list = []
    B1Tbar_list = []
    B2Tbar_list = []
    
    for i in range(len(energies)):

        energy = energies[i]

        BTbar,dE_BTbar,dE2_BTbar = DE2_BarTbar(energy,Hamiltonian0,
               GammaLP,GammaLM,
               GammaR,U,
               nPlist,nMlist,muL,muR,betaL,betaR)


        BTbar_list.append(BTbar)

        B1Tbar_list.append(dE_BTbar)
        B2Tbar_list.append(dE2_BTbar)






    energies_zero_der = func_deriv_zero(energies,B1Tbar_list)
    energies_zero_der2 = func_deriv_zero(energies,B2Tbar_list)
        
    return energies_zero_der,energies_zero_der2






