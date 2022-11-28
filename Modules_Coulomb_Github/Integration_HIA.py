#!/usr/bin/env python
# coding: utf-8

#Author: Karssien Hero Huisman

from matplotlib import pyplot as plt
import numpy as np

## Import negf methods
import negf_HIA_git
import handy_functions_coulomb as hfc




#######################################################################################################################
################################## Calculate Electron Density  ########################################################
#######################################################################################################################


def check_integrand_zero(energies,
                         Hamiltonian0,
                         GammaL,GammaR,
                         U,n00list,
                        muL, muR,
                        betaL,betaR,tol_nintegrand):
    
    '''
    Input:
    - energies = list of energies
    - U = Coulomb interaction strength
    - HamiltonianU : Hamiltonian of scattering region for U finite
    - GammaL,GammaR: Gamma matrices of left,right lead
    - muL,muR = chemical potential of left,right lead
    - betaL, betaR = beta =1/kBT of left, right lead
    - tol_nintegrand = cutoff value on integrand
    Output:
    - check_zerob  = list of round values for lower,upperbound
    - boundbool = boolean. True if integrand is numerically neglible. False if it is not.
    '''
    
    emin,emax = min(energies),max(energies)
    nlist_min =  negf_HIA_git.ndensity_listi(emin,
                                         Hamiltonian0,
                                         GammaL,GammaR,
                                        U,n00list,
                                        muL, muR,
                                        betaL,betaR)


    nlist_max = negf_HIA_git.ndensity_listi(emax,
                                         Hamiltonian0,
                                         GammaL,GammaR,
                                        U,n00list,
                                        muL, muR,
                                        betaL,betaR)



    

    nlist_minmax = hfc.jointwolist(nlist_min,nlist_max)
    

    check_zerob, boundbool = hfc.check_listzero(nlist_minmax,tol_nintegrand)
    
    return check_zerob, boundbool
            


def calc_electron_density_trapz(energies,
                         Hamiltonian0,
                         GammaL,GammaR,
                                U,n00list,
                        muL, muR,
                        betaL,betaR,tol_nintegrand):


    '''
    Input:
    - energies = list of energies
    - listi = list of labels corresponding to orbitals with coulomb interactions
    - U = Coulomb interaction strength
    - Hamiltonian : Hamiltonian of scattering region with Coulomb interactions
    - GammaL,GammaR: Gamma matrices of left,right lead
    - muL,muR = chemical potential of left,right lead
    - betaL, betaR = beta =1/kBT of left, right lead
    Ouput:
    - list of electron densities for site 1,2,3,4...
    '''


    ##### 
    nE00_list = []
    
    shape = Hamiltonian0.shape
    
    check_zerob, boundbool = check_integrand_zero(energies,
                                             Hamiltonian0,
                                             GammaL,GammaR,
                                                  U,n00list,
                                                muL, muR,
                                                betaL,betaR,tol_nintegrand)
    
    if boundbool == True:
        for energy in energies:
            n00_list_energy = negf_HIA_git.ndensity_listi(energy,
                                         Hamiltonian0,
                                         GammaL,GammaR,
                                        U,n00list,
                                        muL, muR,
                                        betaL,betaR)



            nE00_list.append(n00_list_energy)

        #integrate every electron density over all energies.    
        n00list_i = [ np.trapz(np.array(nE00_list)[:,i],energies) for i in range(shape[0])]


        return n00list_i
    
    if boundbool == False:
        
        print('Choose different upper/lower bound for integration, integrand of electron density larger than {}'.format(
                tol_nintegrand))
        
        print(check_zerob)


#######################################################################################################################
##################### Self - Consistent Calculation: arbitrary tol,linear mixing ######################################
#######################################################################################################################



def iteration_calculate_mixing(n00_list,
                        max_iteration ,energies,
                        U,
                        Hamiltonian0,
                        GammaL,GammaR, 
                        muL,muR,betaL, betaR,tol,tol_nintegrand,alpha,plot_bool,trackbool):
    
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
    - alpha = number between [0,1) quantifying "linear mixing". 
    - plot_bool = boolean. True -> <nis>,<c^+is cisbar> are plotted for every iteration.
    - trackbool = boolean. True -> <nis>,<c^+is cisbar> ared saved and plotted after convergence or maximum iteration has been reached.
    '''
    
    
    zero_bool = False
    zero_bool00 =  False
    
    if plot_bool == True:
        trackbool = False
    
    #electron densitylist
    nk_iterations_list= []

    k_list = [i for i in range(max_iteration)]
    acc = int(np.ceil(-np.log(tol)/np.log(10))) 
    
    for k in k_list:

        # Calculate relevant electron densities:
        n00list_new =  calc_electron_density_trapz(energies,
                                                                 Hamiltonian0,
                                                                 GammaL,GammaR,
                                                                        U,n00_list,
                                                                muL, muR,
                                                                betaL,betaR,tol_nintegrand)
        
        

        if k > 0 and plot_bool == True:
            
            nk_iterations_list.append(n00list_new)
            
            if len(nk_iterations_list) > 1 and len(nk_iterations_list) > 1:
                plt.title('nis : Real')
                plt.plot(np.array(nk_iterations_list).real)
                plt.show()
                

        #check if values have converged
        check_zero, zero_bool00 = hfc.check_difference2(n00_list,n00list_new,tol)
        
       
        
    
        print(abs(np.round(check_zero,acc)))
        
        
        if trackbool == True:
            nk_iterations_list.append(n00list_new)
        

        #Stop loop if convergence is achieved
        if zero_bool00 == True:
            
            
            
            if trackbool == True:
                plt.title('nis : Real')
                plt.plot(np.array(nk_iterations_list).real)
                plt.show()
                

            
            
            zero_bool = True
            
            
            nk_iterations_list.append(n00list_new)
            break
            return nk_iterations_list,zero_bool
        
        
            
            
        
        if k == max_iteration-1:
            
            if trackbool == True:
                plt.title('nis : Real')
                plt.plot(np.array(nk_iterations_list).real)
                plt.show()
                
          

   
            nk_iterations_list.append(n00list_new)
            return nk_iterations_list,zero_bool

        #Re-assign electrond densities for next loop
        if zero_bool00 == False:
            
            n00_list = np.add( np.multiply(1-alpha,n00list_new), np.multiply(alpha,n00_list))

        
    return nk_iterations_list,zero_bool



def self_consistent_trapz_mixing_in(V_list,Vmax,
                                  n00_list_guess,
                                  max_iteration,
                                ef,
                                U,
                                Hamiltonian0,
                                GammaL,GammaR, 
                                betaL, betaR,tol,energies,tol_nintegrand,alpha,plot_bool=False,trackbool=False):
    
    '''Input
    - V_list = list of voltages
    - Vmax = maximum voltage
    - max_iteration : maximum number of iterations
    - n00_list_guess = intitial guesses for electron densities.
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
    dim = Hamiltonian0.shape[0]
    klist = hfc.func_list_i(Hamiltonian0)

    for i in range(len(V_list)):
        V = V_list[i]
        muL,muR = ef +V/2, ef -V/2
        
        
        print ('--- V = {} ---'.format(V))
        if i ==0:
            n00_list_init = n00_list_guess
            

              

        if i !=0:
                
            n00_list_init = n_list[i-1] #initial guess
            
            
            
        
        
        nlist_k,zero_bool = iteration_calculate_mixing(n00_list_init,
                                                    max_iteration ,energies,
                                                    U,
                                                    Hamiltonian0,
                                                    GammaL,GammaR, 
                                                    muL,muR,betaL, betaR,tol,tol_nintegrand,alpha,plot_bool,trackbool)

        
        n_list.append(nlist_k[-1])
        convglist.append(zero_bool)
        
        
    return n_list,convglist




def self_consistent_trapz_PN(V_list_pos_bias,Vmax,
                                  max_iteration,
                                ef,
                                U,
                                Hamiltonian0,
                                GammaL,GammaR, 
                                betaL, betaR,tol,
                                energies,tol_nintegrand,alpha,plot_bool=False,trackbool=False):
    
    dim = Hamiltonian0.shape[0]
    pos_bool = hfc.check_list_smaller(np.sign(V_list_pos_bias),1)
    V_list_neg_bias = -1*V_list_pos_bias
    
    
    
    #V_list_pos_bias must be positive bias voltage.
    if pos_bool == True:
        
        #Intial guess for V = 0
        n00_V0_guess = hfc.list_halves(Hamiltonian0)
        
        
        
        
        #Start in equilibrium V = 0
        nV0_list,convglistV0 = self_consistent_trapz_mixing_in([0],0,
                                  n00_V0_guess,
                                  max_iteration,
                                ef,
                                U,
                                Hamiltonian0,
                                GammaL,GammaR, 
                                betaL, betaR,tol,energies,tol_nintegrand,alpha,plot_bool,trackbool)
        
        
        #Sweep for positive and negative bias voltages seperately.
        n00_V_guess  =nV0_list[0]

        n_list_VP,convglist_VP = self_consistent_trapz_mixing_in(V_list_pos_bias,Vmax,
                                  n00_V_guess,
                                  max_iteration,
                                ef,
                                U,
                                Hamiltonian0,
                                GammaL,GammaR, 
                                betaL, betaR,tol,energies,tol_nintegrand,alpha,plot_bool,trackbool)

        n_list_VMprime,convglist_VMprime = self_consistent_trapz_mixing_in(V_list_neg_bias,Vmax,
                                  n00_V_guess,
                                  max_iteration,
                                ef,
                                U,
                                Hamiltonian0,
                                GammaL,GammaR, 
                                betaL, betaR,tol,energies,tol_nintegrand,alpha,plot_bool,trackbool)
        
        n_list_VM = [n_list_VMprime[-1-i] for i in range(len(V_list_neg_bias))]
        convglist_VM = [ convglist_VMprime[-1-i] for i in range(len(V_list_neg_bias)) ] 

        #Join all densities,booleans and voltages into one list.
        n_list_total = hfc.jointwolist(hfc.jointwolist(n_list_VM,nV0_list),n_list_VP)
        convglist_total = hfc.jointwolist(hfc.jointwolist(convglist_VM,convglistV0),convglist_VP)
        return n_list_total,convglist_total
        

    if pos_bool == False:
        print(pos_bool)
        print('Bias voltages must be larger than zero')







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

