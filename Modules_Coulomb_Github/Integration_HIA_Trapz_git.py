#!/usr/bin/env python
# coding: utf-8

from scipy.integrate import quad
from matplotlib import pyplot as plt
import numpy as np

## Import negf methods
import negf_HIA_git


# ### Handy Functions

# In[5]:


def jointwolist(a,b):
    '''
    Input: list a and list b.
    Output: Two joined lists.
    '''
    ab_list = []
    for element in a:
        ab_list.append(element)
    
    for element in b:
        ab_list.append(element)
        
    return ab_list


        
        
        
def check_difference(lista,listb,tol):

    '''
    Input
    - two lists: lista,listb
    - tol = convergence criterium
    Ouput:
    check_zero = absolute value of the difference between lista,listb rounded by tol
    zero_bool = True if convergence is achieved'''
    acc = int(np.ceil(-np.log(tol)/np.log(10))) 
    check_zero = abs(np.round(np.subtract(lista,listb),acc))


    dim = len(check_zero)
    teller = 0


    for element in check_zero:


        if element == 0.0:
            teller +=  1

    if teller == dim:
        zero_bool = True

    if teller != dim:
        zero_bool = False
        
    return check_zero, zero_bool
            


def func_list_i(Hamiltonian):
    '''
    Input:
    -  N X N Hamiltonian
    Output:
    list of labels corresponding the the shape of the shape: [0,1,2,...,N-1]
    '''
    number_sites_spin,number_sites_spin = Hamiltonian.shape
    
    ilist = [i   for i in range(number_sites_spin)]
    
    return ilist


def nlist0_helicene(Hamiltonian):
    
    ''' Input:
    - N X N Hamiltonian
    Output:
    - list: = [1/2,1/2,....] length of the list is N.
    '''
    
    number_sites_spin,number_sites_spin = Hamiltonian.shape

    n_list_s = [0.5   for i in range(number_sites_spin)]
    
    return n_list_s

# In[49]:

def pairwise_swap(xlist):
    '''
    Input:
    list with elements : [a1,a2,b1,b2,...]
    Ouput:
    list where every 2 elements are swapped: [a2,a1,b2,b1,...]
    '''
    
    xlist_swapped = []
    
    for i in range(0,len(xlist),2):
        element_even = xlist[i]
        element_odd = xlist[i+1]
        
        xlist_swapped.append(element_odd)
        xlist_swapped.append(element_even)
    
    return xlist_swapped

# In[49]:





########### trapzoid, self consistency #########

def calc_electron_density_trapz(energies,list_i,
                         Hamiltonian0,
                         GammaL,GammaR,
                        U,nlist,
                        muL, muR,
                        betaL,betaR):


    '''
    Input:
    - energies = list of energies
    - listi = list of labels corresponding to orbitals with coulomb interactions
    - U = Coulomb interaction strength
    - Hamiltonian0 : Hamiltonian of scattering region for U=0
    - GammaL,GammaR: Gamma matrices of left,right lead
    - muL,muR = chemical potential of left,right lead
    - betaL, betaR = beta =1/kBT of left, right lead
    Ouput:
    - list of electron densities for site 1,2,3,4...
    '''


    ##### 
    nE_list = []

    #Calculate densities ni in list_i_ppi given energy from energies
        
    for energy in energies:
        n_ilist = negf_HIA_git.ndensity_listi(energy, list_i,
                                         Hamiltonian0,
                                         GammaL,GammaR,
                                        U,nlist,
                                        muL, muR,
                                        betaL,betaR)



        nE_list.append(n_ilist)

    #integrate every electron density over all energies.    
    nlist_i = [ np.trapz(np.array(nE_list)[:,i],energies) for i in range(len(list_i))]
    
    return nlist_i



def iteration_calculate(n0_list,max_iteration ,energies,
                        U,
                        Hamiltonian0,
                        GammaL,GammaR, 
                        muL,muR,betaL, betaR,tol):
    
    '''Input
    - n0_list : list of electron densities 
    - max_iteration : maximum number of iterations
    - energies : list of energies (to integrate Glesser function over)
    - U = Coulomb interaction strength
    - Hamiltonian0 : Hamiltonian of scattering region for U=0
    - GammaL,GammaR: Gamma matrices of left,right lead
    - muL,muR = chemical potential of left,right lead
    - betaL, betaR = beta =1/kBT of left, right lead
    Ouput:
    - list of electron densities for every iteration
    '''
    
    
    zero_bool = False
    #electron densitylist
    nk_iterations_list = []

    list_i = func_list_i(Hamiltonian0)
    k_list = [i for i in range(max_iteration)]
    
    for k in k_list:
       

        # Calculate relevant electron densities:
        n_new_list =  calc_electron_density_trapz(energies,list_i,
                         Hamiltonian0,
                         GammaL,GammaR,
                        U,n0_list,
                        muL, muR,
                        betaL,betaR)
#         nk_iterations_list.append(n_new_list)
        
#         if k > 0:
#             plt.plot(nk_iterations_list)
#             plt.show()

        check_zero, zero_bool = check_difference(n0_list,n_new_list,tol)
        
        print(check_zero)
        
        if zero_bool == True:
            nk_iterations_list.append(n_new_list)
            break
            return nk_iterations_list,zero_bool
        
        if k == max_iteration-1:
            nk_iterations_list.append(n_new_list)
            return nk_iterations_list,zero_bool

        #Re-assign electrond densities for next loop
        if zero_bool == False:
            
            n0_list = n_new_list
            
     
            
        
    return nk_iterations_list,zero_bool





#################### Analysis of difference ####################


def self_consistent_trapz(V_list,Vmax,
                                  max_iteration,
                                  ef,
                                U,
                                Hamiltonian0,
                                GammaL,GammaR, 
                                betaL, betaR,tol,energiesreal):
    
    '''Input
    - V_list = list of voltages
    - Vmax = maximum voltage
    - max_iteration : maximum number of iterations
    - ef = fermi energy 
    - U = Coulomb interaction strength
    - Hamiltonian0 : Hamiltonian of scattering region for U=0
    - GammaL,GammaR: Gamma matrices of left,right lead
    - muL,muR = chemical potential of left,right lead
    - betaL, betaR = beta =1/kBT of left, right lead
    - energies : list of energies (to integrate Glesser function over)
    Ouput:
    - list of self-consistently calculated electron densities for every voltage
    '''
    
    n_list = []
    convglist = []
    

    for i in range(len(V_list)):

        V = V_list[i]
        print ('--- V = {} ---'.format(V))
        if i ==0:
            n0_list = nlist0_helicene(Hamiltonian0) #initial guess

        if i !=0:
            n0_list = n_list[i-1] #initial guess


        muL,muR = ef +V/2, ef -V/2
        
        
        
        
        nlist_k,zero_bool = iteration_calculate(n0_list,max_iteration ,energiesreal,
                        U,
                        Hamiltonian0,
                        GammaL,GammaR, 
                        muL,muR,betaL, betaR,tol)


        n_list.append(nlist_k[-1])
        convglist.append(zero_bool)
        
        
    return n_list,convglist








def converged_lists(V_list_total,
                      nP_list_total ,convglistP,
                    nM_list_total, convglistM):
    
    '''Input:
    - V_list_total = list of bias voltages
    - nP_list_total,nM_list_total = electron densities for positive,negative magnetization as function of bias voltage
    - convglistP,convglistM = convergence boolean for positive,negative magnetization as function of bias voltage
    Ouput
    - list of electron densities (positive,negative magnetization) that have converged for the corresponding voltages
    - V_list_combined = list of voltages for which positive,negative magnetizatoin electron densities both have converged 
    '''
    
   
    nP_list_combined = []
    nM_list_combined = []
    V_list_combined = []

    for i in range(len(V_list_total)):
        V = V_list_total[i]
        boolP = convglistP[i]
        boolM = convglistM[i]
        
        if boolP == True and boolM == True:
            V_list_combined.append(V)
            nP_list_combined.append(nP_list_total[i])
            nM_list_combined.append(nM_list_total[i])
            
            
    return V_list_combined,nP_list_combined,nM_list_combined      




############ Plot function density of states ############
        
def plot_DOS(V_list,n_list_total,
             ef,U,
             Hamiltonian0,GammaL,GammaR,
             Vmax_plot,plot_chem=False):
    '''
    Input
    - V_list = list of voltages
    - n_list_total = list of electron densities as function of voltage
    - ef = fermi energy
    - U = Coulomb interaction strength
    - Hamiltonian0 = Hamiltonian without coulomb interactions U=0
    - GammaL,GammaR = Gamma matrices of left,right lead in WBL
    - Vmax_plot = maximum voltage for which we should DOS
    Output:
    - Density of states plotted for every voltage
    '''
    if Vmax_plot ==0:
        energies = np.linspace(ef-2,ef+2,2000)
    if Vmax_plot !=0:
        energies = np.linspace(ef-Vmax_plot/2,ef+Vmax_plot/2,2000)
        
    


    
    for i in range(len(V_list)):

    
        V = V_list[i]
        n_list = n_list_total[i]


        muL,muR = ef  + V/2  ,ef-V/2 

   
    



        if V >= 0 and V <= Vmax_plot:
            DOS_list = [negf_HIA_git.density_of_states(energy ,Hamiltonian0,GammaL,GammaR,
                        U,n_list 
                            ).real for energy in energies ] 
            
            plt.plot(energies,DOS_list)
        
            plt.axvline(ef ,color = 'black',label = 'ef')
            plt.ylabel('DOS')
            plt.xlabel('energy [eV]')
                       
            if plot_chem == True:
                plt.axvline(U/2 ,color = 'orange',label = 'U/2')
                plt.axvline(muL,color='red',label = 'muL')
                plt.axvline(muR,color='gold',label = 'muR')
            plt.legend()
            plt.show()

