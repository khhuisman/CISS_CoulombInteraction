#!/usr/bin/env python
# coding: utf-8

#Author: Karssien Hero Huisman

from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import quad



#beta
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
        
        
        if energy != mui:
            fd = (2*beta**2)*((np.sinh(0.5*beta*(energy-mui))/np.sinh(beta*(energy-mui)))**3)*(np.sinh(0.5*beta*(energy-mui)))
            return fd
        if energy == mui:
        
            return 0    
    
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
        
        
        if energy != mui:
            fd = mui_prime*(2*beta**2)*((np.sinh(0.5*beta*(energy-mui))/np.sinh(beta*(energy-mui)))**3)*(np.sinh(0.5*beta*(energy-mui)))
            return fd
        if energy == mui:
        
            return 0
    
    
    
def pairwise_swap(xlist):
    
    xlist_swapped = []
    
    for i in range(0,len(xlist),2):
        element_even = xlist[i]
        element_odd = xlist[i+1]
        
        xlist_swapped.append(element_odd)
        xlist_swapped.append(element_even)
    
    return xlist_swapped   


# Calculate inverse of matrix M:
def inverse(M):
    invers_M = np.array(np.linalg.inv(M),dtype = complex)
    return invers_M

#Advandced, Retarded Green's Function, with WBL approximation.

def GRA_HIA(energy, 
        H,
        GammaL,
        GammaR,
         U,n_list):
    
    '''Input
    - energy: energy
    - H: Hamiltonian matrix for U = 0
    - GammaR/GammaL: imaginary part of self energy matrix in WBL
    - U = Hubbard interaction strength
    - n_list = [n1u,n1d,n2u,n2d,....]
    
    Output:
    
    - Retarded,Advanced Green's Function in the Hubbard I approximation in WBL.
    
    '''
    
    def GP_HIA(energy, 
        H,
        GammaL,
        GammaR,
         U,n_list):
        
        '''
        Returns
        - Advanced Green's function
        '''
        # Hubbard one approximation Retarded Green's function:
        # GR =
        # [    (E - ES - Umat) (E - H0 + (i/2)Gamma) - nU (t + v + (i/2)Gamma)]^-1  [ E-ES - Umat(1-n)]
        # Hubbard one approximation Retarde Green's function:
        # GA =
        # [ E-ES - Umat(1-n)] [ (E - H0 - (i/2)Gamma)(E - ES - Umat) - (t + v + (i/2)Gamma) nU ]^-1  
        #
        # ES = diagonal matrix with onsite energies
        # Umat = diagonal matrix with U on the diagonal
        # H0 = Full Hamiltonian
        # SigmaR = retarded self energy
        # t,v = hopping matrix and spin-orbit coupling hopping matrix respectively.
        # nU = diagonal matrix: U ni\bars \delta_ij \delta_{s,sp}
        
        n_swapped = pairwise_swap(n_list)               # = [n1d,n1u,n2d,n2u,....] densities are swapped per site.
        n_list_cor =  n_swapped

        h_dim = np.shape(H)[1]

        En = np.array(energy*np.identity(h_dim),dtype = complex)

        ES = np.diag([ H[i,i] for i in range(h_dim) ])
        Umat = U*np.identity(h_dim)
        nmat = U*np.diag(n_list_cor) #Un

        EH = np.subtract(En,H)
        Gamma = 0.5*1j*np.array(np.add(GammaL,GammaR),dtype = complex) # 1/2 1j * (GammaL + GammaR)




        En1 = np.subtract(En, np.add(ES,Umat))   # (E - ES - Umat)
        En2 = np.subtract(En, np.add( H, Gamma)) # (E - H0 - i/2 Gamma)
        X   = np.add(np.subtract(H, ES), Gamma)  # (t + v + i/2 Gamma)
        En3 = np.dot(X,nmat) # (t + v + i/2 Gamma)Un

        Noemer = np.subtract(np.dot(En2,En1),En3)  # [ (E - H0 - i/2 Gamma)(E - ES - Umat) - (t + v + i/2 Gamma)Un]
        G_inv = inverse(Noemer)
        teller = np.add(np.subtract(En,np.add(ES,Umat)), nmat)


        #Advanced Green's function
        GA = np.dot(teller,G_inv) #G-
        
        
        return GA
    
    

    #Advanced Green's function
    G_A = GP_HIA(energy, 
        H,
        GammaL,
        GammaR,
         U,n_list)
    #Retarded Greensfunction
    G_R = np.conjugate(np.transpose(G_A)) 

    
  
    return [G_R,G_A]


def dE_GP_HIA(energy, 
        H,
        GammaL,
        GammaR,
         U,n_list):
    
    '''
    Input:
    - energy: energy
    - H: Hamiltonian matrix for U = 0
    - GammaR/GammaL: imaginary part of self energy matrix in WBL
    - U = Hubbard interaction strength
    - n_list = [n1u,n1d,n2u,n2d,....]
    Output:
    First derivative w.r.t. energy of retarded,advanced Green's function
    '''
    
    # Hubbard 1 approximation Green's function:
    # GR =
    # [ 
    #                         (E - ES - Umat) (E - H0 - (i/2)Gamma)
    #                    - nU (t + v + (i/2)Gamma)
    #                   ]^-1 X [ E-ES - Umat(1-n)]
    #
    # ES = diagonal matrix with onsite energies
    # Umat = diagonal matrix with U on the diagonal
    # H0 = Full Hamiltonian
    # SigmaR = retarded self energy
    # t,v = hopping matrix and spin-orbit coupling hopping matrix respectively.
    # nU = diagonal matrix: U ni\bars \delta_ij \delta_{s,sp}

    n_swapped = pairwise_swap(n_list)               # = [n1d,n1u,n2d,n2u,....] densities are swapped per site.
    n_list_cor =  n_swapped

    h_dim = np.shape(H)[1]

    En = np.array(energy*np.identity(h_dim),dtype = complex)

    ES = np.diag([ H[i,i] for i in range(h_dim) ])
    Umat = U*np.identity(h_dim)
    nmat = U*np.diag(n_list_cor) #Un

    EH = np.subtract(En,H)
    Gamma = 0.5*1j*np.array(np.add(GammaL,GammaR),dtype = complex) # 1/2 1j * GammaL + GammaR




    En1 = np.subtract(En, np.add(ES,Umat))   # (E - ES - Umat)
    En2 = np.subtract(En, np.add( H, Gamma)) # (E - H0 - i/2 Gamma)
    X   = np.add(np.subtract(H, ES), Gamma)  # (t + v + i/2 Gamma)
    En3 = np.dot(nmat,X) # Un(t + v + i/2 Gamma)

    Noemer = np.subtract(np.dot(En1,En2),En3)
    Noemerp = np.subtract(np.dot(En2,En1),En3)
    G_inv = inverse(Noemer)
    G_invp = inverse(Noemerp)
    
    teller = np.add(np.subtract(En,np.add(ES,Umat)), nmat) 


    #Retarded, advanced Green's function
    GA = np.dot(G_inv,teller) #G+

       
    C = np.add(En2, En1)
    
    
    #Retarded, advanced Green's function
    dEGA = np.dot(G_inv,np.subtract(
                       np.identity(h_dim), 
                       np.dot(C,GA)
                      )
                 )
                  
    dEGR = np.conjugate(np.transpose(dEGA))
    

    return dEGA,dEGR 



def dE2_GP_HIA(energy, 
        H,
        GammaL,
        GammaR,
         U,n_list):
    
    '''
    Input:
    - energy: energy
    - H: Hamiltonian matrix for U = 0
    - GammaR/GammaL: imaginary part of self energy matrix in WBL
    - U = Hubbard interaction strength
    - n_list = [n1u,n1d,n2u,n2d,....]
    Output:
    Second derivative w.r.t. energy of retarded,advanced Green's function
    '''
    
    # Hubbard 1 approximation Green's function:
    # GR =
    # [ 
    #                         (E - ES - Umat) (E - H0 - (i/2)Gamma)
    #                    - nU (t + v + (i/2)Gamma)
    #                   ]^-1 X [ E-ES - Umat(1-n)]
    #
    # ES = diagonal matrix with onsite energies
    # Umat = diagonal matrix with U on the diagonal
    # H0 = Full Hamiltonian
    # SigmaR = retarded self energy
    # t,v = hopping matrix and spin-orbit coupling hopping matrix respectively.
    # nU = diagonal matrix: U ni\bars \delta_ij \delta_{s,sp}

    n_swapped = pairwise_swap(n_list)               # = [n1d,n1u,n2d,n2u,....] densities are swapped per site.
    n_list_cor =  n_swapped

    h_dim = np.shape(H)[1]

    En = np.array(energy*np.identity(h_dim),dtype = complex)

    ES = np.diag([ H[i,i] for i in range(h_dim) ])
    Umat = U*np.identity(h_dim)
    nmat = U*np.diag(n_list_cor) #Un

    EH = np.subtract(En,H)
    Gamma = 0.5*1j*np.array(np.add(GammaL,GammaR),dtype = complex) # 1/2 1j * GammaL + GammaR




    En1 = np.subtract(En, np.add(ES,Umat))   # (E - ES - Umat)
    En2 = np.subtract(En, np.add( H, Gamma)) # (E - H0 - i/2 Gamma)
    X   = np.add(np.subtract(H, ES), Gamma)  # (t + v + i/2 Gamma)
    En3 = np.dot(nmat,X) # Un(t + v + i/2 Gamma)

    Noemer = np.subtract(np.dot(En1,En2),En3)
    G_inv = inverse(Noemer)
    
#     Noemerp = np.subtract(np.dot(En2,En1),En3)
#     G_invp = inverse(Noemerp)
    
    teller = np.add(np.subtract(En,np.add(ES,Umat)), nmat) 


    #Retarded, advanced Green's function
    GA = np.dot(G_inv,teller) #G+
    GR = np.conjugate(np.transpose(GA))
       
    C = np.add(En2, En1)
    
    
    #Retarded, advanced Green's function
    dEGA = np.dot(G_inv,np.subtract(
                       np.identity(h_dim), 
                       np.dot(C,GA)
                      )
                 )
                  
    dEGR = np.conjugate(np.transpose(dEGA))
    
    dE2_GA = np.multiply(-2,np.dot(G_inv,
                           np.add(np.dot(C,dEGA),
                                  GA)
                                 )
                        )
    
    dE2_GR = np.conjugate(np.transpose(dE2_GA))
    return GA,GR,dEGA,dEGR,dE2_GA,dE2_GR


#########################################################################################################
############################# Transmission for strictly 2-Terminal systems ##############################
#########################################################################################################


#Transmission left to right
# Only valid for 2terminal junctions
def TLR(energy,H
        ,GammaL,GammaR,U,n_list ):
    
    '''
    Input:
    - energy: energy
    - H: Hamiltonian matrix for U = 0
    - GammaR/GammaL: imaginary part of self energy matrix in WBL
    - U = Hubbard interaction strength
    - n_list = [n1u,n1d,n2u,n2d,....]
    Output:
    Transmission from left to right lead
    '''
    GR,GA = GRA_HIA(energy, 
                H,
                GammaL,
                GammaR,
                 U,n_list)
    
    T = np.dot(np.dot(np.dot(GammaL,GA),GammaR),GR)
    T_LR = np.matrix.trace(T).real
    
    
    return T_LR

#Transmission right to left
# Only valid for 2terminal junctions
def TRL(energy, 
                H,
                GammaL,
                GammaR,
                 U,n_list
        ):
    
    '''
    Input:
    - energy: energy
    - H: Hamiltonian matrix for U = 0
    - GammaR/GammaL: imaginary part of self energy matrix in WBL
    - U = Hubbard interaction strength
    - n_list = [n1u,n1d,n2u,n2d,....]
    Output:
    Transmission from right to left lead
    '''
    
    GR,GA = GRA_HIA(energy, 
                H,
                GammaL,
                GammaR,
                 U,n_list)
    
    T = np.dot(np.dot(np.dot(GammaR,GA),GammaL),GR)
    T_RL = np.matrix.trace(T).real
    return T_RL

############ 
def dETLR(energy,H
        ,GammaL,GammaR,U,n_list ):
    
    
    GR,GA = GRA_HIA(energy, 
                H,
                GammaL,
                GammaR,
                 U,n_list)
                  
                  
                  
    dEGA,dEGR = dE_GP_HIA(energy, 
        H,
        GammaL,
        GammaR,
         U,n_list)
    
    T1 = np.dot(np.dot(np.dot(GammaL,dEGA),GammaR),GR)
    T2 = np.dot(np.dot(np.dot(GammaL,GA),GammaR),dEGR)
    
                  
    T_LR = np.matrix.trace(np.add(T1,T2)).real
    
    
    return T_LR




def dEn_TLR(energy,H
        ,GammaL,GammaR,U,n_list ):
    
    
   
                  
    #Calculate Green's functions             
    GA,GR,dEGA,dEGR,dE2GA,dE2GR = dE2_GP_HIA(energy, 
        H,
        GammaL,
        GammaR,
         U,n_list)
    
    #TLR
    T = np.dot(np.dot(np.dot(GammaL,GA),GammaR),GR)
    TLR = np.matrix.trace(T).real
    
    #dETLR
    dET1 = np.dot(np.dot(np.dot(GammaL,dEGA),GammaR),GR)
    dET2 = np.dot(np.dot(np.dot(GammaL,GA),GammaR),dEGR)
    
                
    dET_LR = np.matrix.trace(np.add(dET1,dET2)).real
    
    #dE2TLR
    dE2T1 = np.dot(np.dot(np.dot(GammaL,dE2GA),GammaR),GR)
    dE2T2 = np.multiply(2,np.dot(np.dot(np.dot(GammaL,dEGA),GammaR),dEGR))
    dE2T3 = np.dot(np.dot(np.dot(GammaL,GA),GammaR),dE2GR)
    
                  
    dE2T_LR = np.matrix.trace(np.add(np.add(dE2T1,dE2T2),dE2T3)).real
    
    
    return TLR, dET_LR,dE2T_LR



#########################################################################################################
############################### Quantities related to Electron Densities  ###############################
#########################################################################################################

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
                U,n_list,
                muL, muR,
                betaL,betaR):
    
    GR,GA = GRA_HIA(energy, 
                H,
                GammaL,
                GammaR,
                 U,n_list)
    
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
                U,n_list,
                muL, muR,
                betaL,betaR):
    
    GR,GA = GRA_HIA(energy, 
                H,
                GammaL,
                GammaR,
                 U,n_list)
    
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
                U,n_list,
                    ):
    

    
    GR,GA = GRA_HIA(energy, 
                H,
                GammaL,
                GammaR,
                 U,n_list)
    
    A = 1j*np.subtract(GR,GA)
    
    DOS = np.matrix.trace(A)/(2*np.pi) 
    
    return DOS





def ndensity_listi(energy, ilist,
                 H,
                 GammaL,GammaR,
                U,n_list,
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
                U,n_list,
                muL, muR,
                betaL,betaR)
    
    n_ilist = [ np.real(np.multiply(-1j,Gless[i,i]))/(2*np.pi) for i in ilist]
    
    
    return n_ilist







##############################################################################################################################
##############################################################################################################################
####                                               ####
#### Integrand for calculating 2 terminal current  ####
####                                               ####
##############################################################################################################################
##############################################################################################################################



def integrand_current_HIA(energy, 
                          Hamiltonian0 ,
                          GammaL,GammaR,
                          U,n_list, 
                          betaL,betaR,muL,muR ):
    
    '''
    Input:
    - energy = energy of electron in eV.
    - Hamiltonian0 = Hamiltonian without interactions (U=0)
    - GammaL,GammaR = Gamma matrices of left and right lead respectively in the wide band limit
    - U       = Coulomb interaction strength
    - n_list  = electron densities for every site as a function of voltage.
    - betaL,betaR =  beta  [1/(kB T)] of the left,right lead in eV
    - muL,muR = chemical potential of left,right lead
    

    Output:
    - Transmission weigthed with fermi-dirac functions for a system in the Hubbard One Approximation'''
        
    
    fL = fermi_dirac(energy,muL,betaL)
    fR = fermi_dirac(energy,muR,betaR)
    
    integrand = TLR(energy,Hamiltonian0,GammaL,GammaR,U,n_list )*(fL-fR)
        
    return integrand


def integrand_deltaI_HIA(energy, 
                          Hamiltonian0 ,
                          GammaLP,GammaR,
                          GammaLM,
                          U,nP_list, nM_list,
                          betaL,betaR,muL,muR ):
    
    '''
    Input:
    - energy = energy of electron in eV.
    - Hamiltonian0 = Hamiltonian without interactions (U=0)
    - GammaLP,GammaLM = Gamma matrices of left lead positive and negative magnetic polarization respectively in WBL
    - GammaR = Gamma matrix right in the wide band limit
    - U       = Coulomb interaction strength
    - nP_list,nM_list  = electron densities for positive and negative magnetisation respectively 
    - betaL,betaR =  beta  [1/(kB T)] of the left,right lead in eV
    - muL,muR = chemical potential of left,right lead
    
        
    
    Output:
    - Integrand of magnetocurrent calculated with Landauer-Buttiker Formula for a system in the Hubbard One Approximation'''
        
    
    fL = fermi_dirac(energy,muL,betaL)
    fR = fermi_dirac(energy,muR,betaR)
    
    integrand = (TLR(energy,Hamiltonian0,GammaLP,GammaR,U,nP_list )-TLR(energy,Hamiltonian0,GammaLM,GammaR,U,nM_list ))*(fL-fR)
        
    return integrand

def integrand_barI_HIA(energy, 
                          Hamiltonian0 ,
                          GammaLP,GammaR,
                          GammaLM,
                          U,nP_list, nM_list,
                          betaL,betaR,muL,muR ):
    
    '''
    Input:
    - energy = energy of electron in eV.
    - Hamiltonian0 = Hamiltonian without interactions (U=0)
    - GammaLP,GammaLM = Gamma matrices of left lead positive and negative magnetic polarization respectively in WBL
    - GammaR = Gamma matrix right in the wide band limit
    - U       = Coulomb interaction strength
    - nP_list,nM_list  = electron densities for positive and negative magnetisation respectively 
    - betaL,betaR =  beta  [1/(kB T)] of the left,right lead in eV
    - muL,muR = chemical potential of left,right lead
    
        
    
    Output:
    - Sum of integrand for positive,negative magnetisation for a system in the Hubbard One Approximation'''
        
    
    fL = fermi_dirac(energy,muL,betaL)
    fR = fermi_dirac(energy,muR,betaR)
    
    integrand = (TLR(energy,Hamiltonian0,GammaLP,GammaR,U,nP_list )+TLR(energy,Hamiltonian0,GammaLM,GammaR,U,nM_list ))*(fL-fR)
        
    return integrand




