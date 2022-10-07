#!/usr/bin/env python
# coding: utf-8

#Author: Karssien Hero Huisman



from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import quad

import tinyarray

# define Pauli-matrices for convenience
sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j, 0]])
sigma_z = tinyarray.array([[1, 0], [0, -1]])



def func_beta(T):
    kB = 8.617343*(10**(-5)) # eV/Kelvin
    ''' Input:
    -Temperature in Kelvin
    Output:
    - beta in eV^-1
    '''
    if T > 0:
        return 1/(kB*T)
    if T ==0:
        return np.infty
    if T<0:
        print('Temperature cannot be negative')

# Fermi-Dirac function
def fermi_dirac(energy,mui,beta):
    '''Input:
    - energy of the electron
    - mui = chemical potential
    - beta = 1/(kB*T) with T the temperature and kB the Boltzmann constant
    Output:
    - The fermi-Dirac distribution for energy 
    '''
    
    if beta < 0:
        print('Error: Beta must be positive)')
    elif beta == np.infty:
        # return heavi-side function if T=0 
        fd = np.heaviside(-(energy-mui),1) 
        return fd
    else:
        deltae = energy-mui
        if np.sign(deltae) == -1:
            fd = 1/(np.exp(beta*(energy-mui) ) + 1 )
            return fd
        
        if np.sign(deltae) == 1:
            fd = np.exp(beta*(mui-energy) )/(np.exp(beta*(mui-energy) ) + 1 )
            return fd
        
        if np.sign(deltae) == 0:
            fd = 1/(np.exp(beta*(energy-mui) ) + 1 )
            return fd
    
    
    
# Derivative of Fermi-Dirac function
def fermi_prime_dirac(energy,mui,beta,mui_prime):
    '''Input:
    - energy of the electron
    - mui = chemical potential
    - beta = 1/(kB*T) with T the temperature and kB the Boltzmann constant
    Output:
    - The "derivative of the fermi-Dirac w.r.t chemical potential" distribution for energy 
    '''
    
    if beta < 0:
        print('Error: Beta must be positive)')
    if beta >0:

        fd = mui_prime*beta*(1/4)*(1/np.cosh(beta*(energy-mui)/2))**2 
        return fd
    
def fermi_pp_dirac(energy,mui,beta,mui_prime):
    '''Input:
    - energy of the electron
    - mui = chemical potential
    - beta = 1/(kB*T) with T the temperature and kB the Boltzmann constant
    Output:
    - The "derivative of the fermi-Dirac w.r.t chemical potential" distribution for energy 
    '''
    
    if beta < 0:
        print('Error: Beta must be positive)')
    if beta >0:
        
#         if energy != mui:
#             fd = mui_prime*(2*beta**2)*((1/np.sinh(beta*(energy-mui)))**3)*(np.sinh(0.5*beta*(energy-mui))**4)
#             return fd
        
        if energy != mui:
            fd = mui_prime*(2*beta**2)*((np.sinh(0.5*beta*(energy-mui))/np.sinh(beta*(energy-mui)))**3)*(np.sinh(0.5*beta*(energy-mui)))
            return fd
        if energy == mui:
        
            return 0
    
    
def de_fermi_dirac(energy,mui,beta):
    '''Input:
    - energy of the electron
    - mui = chemical potential
    - beta = 1/(kB*T) with T the temperature and kB the Boltzmann constant
    Output:
    - The "derivative of the fermi-Dirac w.r.t chemical potential" distribution for energy 
    '''
    
    if beta < 0:
        print('Error: Beta must be positive)')
    if beta >0:

        fd = -1*beta*(1/4)*(1/np.cosh(beta*(energy-mui)/2))**2 
        return fd
    
def de2_fermi_dirac(energy,mui,beta):
    '''Input:
    - energy of the electron
    - mui = chemical potential
    - beta = 1/(kB*T) with T the temperature and kB the Boltzmann constant
    Output:
    - The "derivative of the fermi-Dirac w.r.t chemical potential" distribution for energy 
    '''
    
    if beta < 0:
        print('Error: Beta must be positive)')
    if beta >0:
        
#         if energy != mui:
#             fd = mui_prime*(2*beta**2)*((1/np.sinh(beta*(energy-mui)))**3)*(np.sinh(0.5*beta*(energy-mui))**4)
#             return fd
        
        if energy != mui:
            fd = (2*beta**2)*((np.sinh(0.5*beta*(energy-mui))/np.sinh(beta*(energy-mui)))**3)*(np.sinh(0.5*beta*(energy-mui)))
            return fd
        if energy == mui:
        
            return 0    
    
    
# Derivative of Fermi-Dirac function
def fermi_prime_dirac_heat(energy,mui,beta):
    '''Input:
    - energy of the electron
    - mui = chemical potential
    - beta = 1/(kB*T) with T the temperature and kB the Boltzmann constant
    Output:
    - The "derivative of the fermi-Dirac w.r.t chemical potential" distribution for energy 
     taken from equation (12) "Thermoelectric Signatures of Coherent Transport in Single-Molecule Heterojunctions,2009"
    '''
    
    if beta < 0:
        print('Error: Beta must be positive)')
    if beta >0:
        fd = (beta/4)*(1/(np.cosh(beta*(energy-mui)/2)**2) )
        return fd



# Calculate inverse of matrix M:
def inverse(M):
    invers_M = np.array(np.linalg.inv(M),dtype = complex)
    return invers_M

#Advandced, Retarded Green's Function, with WBL approximation.

def GRA(energy, 
        H,
        GammaL,
        GammaR
         ):
    
    '''Input
    - en: energy
    - H: Hamiltonian matrix
    - GammaR/GammaL: imaginary part of self energy matrix in WBL
    
    Output:
    
    - Retarded,Advanced Green's Function in WBL.
    
    '''
    
    h_dim = np.shape(H)[1]
    Gamma = 0.5*1j*np.array(np.add(GammaL,GammaR),dtype = complex)
    
    
    En = np.array(energy*np.identity(h_dim),dtype = complex)
    EH = np.subtract(En,H)
    
    #Retarded, advanced Green's function
    G_R = inverse(np.add(EH,Gamma))
    G_A = np.transpose(np.conjugate(G_R))
        
        
 
    return [G_R,G_A]

####### Transmission for strictly 2-Terminal systems  #######



#Transmission left to right
# Only valid for 2terminal junctions
def TLR(energy,H
        ,GammaL,GammaR ):
    
    
    GR,GA = GRA(energy,H,GammaL,GammaR)
    
    T = np.dot(np.dot(np.dot(GammaL,GA),GammaR),GR)
    T_LR = np.matrix.trace(T).real
    
    
    return T_LR

#Transmission right to left
# Only valid for 2terminal junctions
def TRL(energy,H,
        GammaL,GammaR
        ):
    
    
    GR,GA = GRA(energy,H,GammaL,GammaR)
    
    T = np.dot(np.dot(np.dot(GammaR,GA),GammaL),GR)
    T_RL = np.matrix.trace(T).real
    return T_RL


def dETij(energy,H,
        GammaL,GammaR,Xlist):
    
    
    GR,GA = GRA(energy,H,GammaL,GammaR)
    
    
    X = np.diag(Xlist)
    
    bit1 = np.dot(np.dot(X,GR),GammaR)
    bit2 = np.dot(np.dot(GA,GammaR),X)
    
    bits = np.add(bit1,bit2)
    
    T1 = np.matrix.trace(np.dot(np.dot(np.dot(GammaL,GR),bits),GA)).real
    
    
    
    return T1


def dEn_Tij(energy,H,
        GammaL,GammaR):
    
    
    GR,GA = GRA(energy,H,GammaL,GammaR)
    
    T = np.dot(np.dot(np.dot(GammaR,GA),GammaL),GR)
    T_RL = np.matrix.trace(T).real
   
    
    GA2 = np.dot(GA,GA)
    GR2 = np.dot(GR,GR)
    
    GA3 = np.dot(GA,GA2)
    GR3 = np.dot(GR,GR2)
    
    #dETLR
    dETLR1 = np.dot(np.dot(np.dot(GammaR,GA2),GammaL),GR)
    dETLR2 = np.dot(np.dot(np.dot(GammaR,GA),GammaL),GR2)
    dETLR = -1*np.matrix.trace(np.add(dETLR1,dETLR2)).real
    
    
    #dE2TLR
    dE2TLR1 = np.dot(np.dot(np.dot(GammaR,GR),GammaL),GA3)
    dE2TLR2 = np.dot(np.dot(np.dot(GammaR,GR2),GammaL),GA2)
    dE2TLR3 = np.dot(np.dot(np.dot(GammaR,GR3),GammaL),GA)
            
    dE2TLR = 2*np.matrix.trace(np.add( np.add(dE2TLR1,dE2TLR2),dE2TLR3)).real
    
    return T_RL,dETLR,dE2TLR






#########################################################################################################
############################### Quantities related to Electron Densities  ###############################
#########################################################################################################

########### Sigma's ###########

def SigmaLesser(energy,
                GammaL,GammaR,
                muL , muR,
                betaL,betaR):
    
    shape = GammaL.shape
    SigmaLess = np.zeros(shape)
    
    fL = fermi_dirac(energy,muL,betaL)
    fR = fermi_dirac(energy,muR,betaR)
    
    # Equation (2.123) from Seldenthuys thesis:
        #Sigma<_alpha = 1j Gamma_alpha*f_alpha 
        #Sigma<       = Sigma<_L + Sigma<_R
        #Sigma<       = 1j( GammaL*fL+ GammaR*fR)
    
    SigmaLess = np.multiply(1j,
                                np.add(
                                        np.multiply(GammaL,fL), 
                                        np.multiply(GammaR,fR)
                                      )
                           )
    
    return SigmaLess



def SigmaGreater(energy,
                GammaL,GammaR,
                muL , muR,
                betaL,betaR):
    
    
    shape = GammaL.shape
    fL = fermi_dirac(energy,muL,betaL)
    fR = fermi_dirac(energy,muR,betaR)
    
    
    shape_identity = GammaL.shape
#     iden = np.multiply(2,np.identity(shape_identity[0]))
    iden = np.add(GammaL,GammaR)
    
    # alpha = lead label
    # Equation (2.123) from Seldenthuys thesis:
        #Sigma>_alpha = -1j Gamma_alpha(1- f_alpha) 
        #Sigma>       = Sigma>_L + Sigma>_R
        #Sigma>       = -1j (GammaL + GammaR - GammaL*fL - GammaR*fR)
    SigmaGreat = np.multiply(-1j,np.add(iden,
                                       np.add(
                                         np.multiply(GammaL,-fL), 
                                         np.multiply(GammaR,-fR)
                                              )
                                      )
                                
                           )
    
    return SigmaGreat
    
########### Lesser Green's Function ###########


def GLesser(energy,H,
                GammaL,GammaR,
                muL, muR,
                betaL,betaR):
    
    GR,GA = GRA(energy,
                     H,
                     GammaL,GammaR)
    
    SigmaLess = SigmaLesser(energy,GammaL,GammaR,
                            muL, muR ,
                            betaL,betaR)
    
    
    
    
    # G< = GR.Sigma<.GA
    Gless = np.dot(GR,
                        np.dot(SigmaLess,GA)
                     )
    
    return Gless



########### Greater Green's Function ###########

def GGreater(energy,H,
                GammaL,GammaR,
                muL, muR,
                betaL,betaR):
    
    GR,GA = GRA(energy,
                     H,
                     GammaL,GammaR)
    
    SigmaGreat = SigmaGreater(energy,GammaL,GammaR,
                            muL, muR ,
                            betaL,betaR)
    
    
    
    
    # G< = GR.Sigma<.GA
    GG = np.dot(GR,
                        np.dot(SigmaGreat,GA)
                     )
    
    return GG


########### Density of states ###########

def density_of_states(energy,H,GammaL,GammaR,
                muL, muR,
                betaL,betaR):
    
    GR,GA = GRA(energy,
                     H,
                     GammaL,GammaR)
    
    A = 1j*np.subtract(GR,GA)
    
    DOS = np.matrix.trace(A)/(2*np.pi) 
    
    return DOS



########### Electron densities ###########


def ndensityi(energy, i,
                 H,
                 GammaL,GammaR,
                muL, muR,
                betaL,betaR):
    
    '''
    Input:
    - energy = energy
    - i = index of a site/spin
    - GammaL,GammR = Gamma Matrices of left,right lead.
    - muL,muR = chemical potential of the left,right lead.
    - betaL,betaR = beta = (kBT)^-1 of left,right lead.
    Ouput
    - Electron density <ni> for index i
    '''
    
    
    Gless = GLesser(energy,H,GammaL,GammaR,
                muL, muR,
                betaL,betaR)
    
    ni_e = np.real(np.multiply(-1j,Gless[i,i]))/(2*np.pi)
    
    
    return ni_e



def ndensity_listi(energy, ilist,
                 H,
                 GammaL,GammaR,
                muL, muR,
                betaL,betaR):
    
    
    '''
    Input:
    - energy = energy
    - ilist = list of indices for which one want to calculate the electron density.
    - GammaL,GammR = Gamma Matrices of left,right lead.
    - muL,muR = chemical potential of the left,right lead.
    - betaL,betaR = beta = (kBT)^-1 of left,right lead.
    Ouput
    - List of electron densities on site i with given a certain energy.
    '''
    
    
    Gless = GLesser(energy,H,GammaL,GammaR,
                muL, muR,
                betaL,betaR)
    
    n_ilist = [ np.real(np.multiply(-1j,Gless[i,i]))/(2*np.pi) for i in ilist]
    
    
    return n_ilist


##############################################################################
#### Integrand for calculating 2 terminal current  ##########################
#############################################################################



def integrand_current(energy, H,
                             GammaL,GammaR,
                          betaL,betaR,muL,muR):
    
    '''
    Input:
    System paramters (see ChiralChainModel.py for description of all)
    - energy of incoming electron.
    - betaL,betaR = the beta = 1/(kB T) of the left,right lead
    - muL,muR = chemical potential of left,right lead

    Output:
    - Current calculated with Landauer-Buttiker Formula '''
        
    
    fL = fermi_dirac(energy,muL,betaL)
    fR = fermi_dirac(energy,muR,betaR)
    
    integrand = TLR(energy,H,GammaL,GammaR )*(fL-fR)
    
    return integrand


def integrand_current_deltaI(energy, 
                             HP,HM,
                             GammaLP,GammaLM,
                             GammaR,
                          betaL,betaR,muL,muR):
    
    '''
    Input:
    System paramters (see ChiralChainModel.py for description of all)
    - energy of incoming electron.
    - betaL,betaR = the beta = 1/(kB T) of the left,right lead
    - muL,muR = chemical potential of left,right lead
    
    
        
    
    Output:
    - Current calculated with Landauer-Buttiker Formula '''
        
    
    fL = fermi_dirac(energy,muL,betaL)
    fR = fermi_dirac(energy,muR,betaR)
    
    integrandP = TLR(energy,HP,GammaLP,GammaR )
    integrandM = TLR(energy,HM,GammaLM,GammaR )
    integrand = (integrandP-integrandM)*(fL-fR)
    
    return integrand


def integrand_leadi_energy_current(energy,mui, H,
                             GammaL,GammaR,
                          betaL,betaR,muL,muR):
    
    '''
    Input:
    System paramters (see ChiralChainModel.py for description of all)
    - mu chemical potential of lead i
    - energy of incoming electron.
    - betaL,betaR = the beta = 1/(kB T) of the left,right lead
    - muL,muR = chemical potential of left,right lead

    Output:
    - Energy current for lead i calculated with Landauer-Buttiker Formula '''
        
    
    fL = fermi_dirac(energy,muL,betaL)
    fR = fermi_dirac(energy,muR,betaR)
    
    integrand = (energy-mui)*TLR(energy,H,GammaL,GammaR )*(fL-fR)
    
    return integrand


def integrand_MW(energy, H,
                             GammaL,GammaR,
                          betaL,betaR,muL,muR):
    
    '''
    Input:
    System paramters (see ChiralChainModel.py for description of all)
    - energy of incoming electron.
    - betaL,betaR = the beta = 1/(kB T) of the left,right lead
    - muL,muR = chemical potential of left,right lead

    Output:
    - Current calculated with the Meir-Wingreen formula (eq. 6 of "Landauer Formula for the Current through an Interacting Electron Region,20/04/92".) '''
    
    
    
    
    fL = fermi_dirac(energy,muL,betaL)
    fR = fermi_dirac(energy,muR,betaR)
    
    G_R,G_A = GRA(energy, 
            H,
            GammaL,
            GammaR
             )    
    
    DeltaG = np.subtract(G_R,G_A)
    
    int1 = (1j/2)*np.matrix.trace( 
                            np.dot(fL*GammaL - fR*GammaR,DeltaG) 
                            )
    
    int2 = (1j/2)*np.matrix.trace(
                            np.dot(GammaL - GammaR,
                                   GLesser(energy,H,
                                            GammaL,GammaR,
                                            muL, muR,
                                            betaL,betaR)
                                    )
    )
    
    
    return (int1+int2)



##############################################################################################################################
##############################################################################################################################
####                                                                ####
#### Integrand for calculating 2 terminal differential conductance  ####
####                                                                ####
##############################################################################################################################
##############################################################################################################################


def integrand_difcon(energy, H,
                             GammaL,GammaR,
                          betaL,betaR,
                         muL,muR,
                    muL_prime,muR_prime):
    
    '''
    Input:
    System paramters (see ChiralChainModel.py for description of all)
    - energy of incoming electron.
    - betaL,betaR = the beta = 1/(kB T) of the left,right lead
    - muL,muR = chemical potential of left,right lead
    
    
        
    
    Output:
    - Differential conductance for a voltage indepedent system calculated with Landauer-Buttiker Formula '''
        
    
    fL_prime = fermi_prime_dirac(energy,muL,betaL,muL_prime)
    fR_prime =fermi_prime_dirac(energy,muR,betaR,muR_prime)
    
    integrand_dc = TLR(energy,H,GammaL,GammaR )*(fL_prime-fR_prime)
    
    return integrand_dc





##############################################################################
#################### Hartree-Fock Hamiltonian ################################
##############################################################################

def pairwise_swap(xlist):
    
    xlist_swapped = []
    
    for i in range(0,len(xlist),2):
        element_even = xlist[i]
        element_odd = xlist[i+1]
        
        xlist_swapped.append(element_odd)
        xlist_swapped.append(element_even)
    
    return xlist_swapped


def nlist0_helicene(Hamiltonian):
    
    ''' Input:
    - Hamiltonian
    Output:
    - List with initial values for the electron densities of helicene.
    '''
    
    number_sites_spin,number_sites_spin = Hamiltonian.shape

    n_list_s = [0.5   for i in range(number_sites_spin)]
    
    return n_list_s


def Hamiltonian_HF(n_list,U,Hamiltonian0):
    
    
    '''Input
    - nlist = list of electron densities in the specific order: [n1u,n1d,n2u,n2d,...]
    - U = Onsite Hubbard parameter
    - Hamiltonian0: hamiltonian for U=0
    Output
    -  Hamiltonian in HF for U =! 0.'''
    
    # HamiltonianU = [<ni\bar{\sigma}> - 1/2 ] \hat{n}_{is}
    n0list = nlist0_helicene(Hamiltonian0) # [0.5, 0.5,....]
    n_list_swapped = pairwise_swap(n_list) # [n1d,n1u,n2d,n2u,...]
    HamiltonianU = np.multiply(U,np.diag(np.subtract(n_list_swapped,n0list)))
    
    # HamiltonianUfull = H0 + HamiltonianU
    HamiltonianUfull = np.add(Hamiltonian0,HamiltonianU)
    
    
    return HamiltonianUfull

