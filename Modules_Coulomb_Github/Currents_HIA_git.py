#!/usr/bin/env python
# coding: utf-8


from scipy.integrate import quad
from matplotlib import pyplot as plt
import numpy as np


import negf_HIA_git # Non - Equilibrium Green's Function method
import HubbardOne_Error_Estimate_git as HIA_error # Module identifies sharp peaks in integrand 


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
        
        

##### Calculate current with trapezoid method ######



# In[ ]:


def calc_current_quad(V_list_total,n_list_total,
                       ef,
                U,Hamiltonian0,
                GammaL,GammaR, betaL,betaR,
                     emin_quad=-np.infty,emax_quad=np.infty):
    
    
    '''
    Input:
    - V_list = list of voltages (positive and negative )
    - n_list_total = list of electron densities of the P_pi orbitals 
    - Hamiltonian0 = Hamiltonianfor U =0
    - GammaL,GammaR: Gamma matrices of left,right lead
    - muL,muR = chemical potential of left,right lead
    - betaL, betaR = beta =1/kBT of left, right lead
    - emin_quad,emax_quad : minimum and maximum bound for integration
    Ouput:
    - Current calculted with the quad scipy method
    '''

    I_list_quad = []
    I_list_error = []

    for i in range(len(V_list_total)):

        V = V_list_total[i]
        print('--- V = {} ---'.format(V))
        #Bias
        muL,muR = ef +V/2, ef -V/2

        #Hamiltonian
        n_list = n_list_total[i]
        
        


        IL = quad(negf_HIA_git.integrand_current_HIA, emin_quad,emax_quad,
             args= (Hamiltonian0 ,GammaL,GammaR,U,n_list, betaL,betaR,muL,muR ) 
            )

        I_list_quad.append(IL[0])
        I_list_error.append(IL[1])

    return I_list_quad,I_list_error



def calc_deltaI_quad(V_list_total,
                       ef,
                        Hamiltonian0 ,
                          GammaLP,GammaR,
                          GammaLM,
                          U,nP_list_total, nM_list_total,
                          betaL,betaR):
    
    
    '''
    Input:
    - V_list_total = list of voltages (positive and negative )
    - ef = Fermi energy
    - Hamiltonian0 = hamiltonian for U = 0 
    - GammaLP,GammaLM: Gamma matrices of left for positive, negative magnetization
    - GammaR: Gamma matrices of right lead
    - U = onsite Coulomb interaction strength
    - nP_list_total,nM_list_total = list of electron densities for positive, negative magnetization
    - betaL, betaR = beta =1/kBT of left, right lead
    Ouput:
    - List of Delta I for voltage in calculted with the quad scipy method
    - Numerical Error estimate on Delta I, as given by "quad function"
    '''

    dI_list_quad = []
    dI_list_error = []

    for i in range(len(V_list_total)):

        V = V_list_total[i]
        print('--- V = {} ---'.format(V))
        #Bias
        muL,muR = ef + V/2, ef -V/2

        #Electron densities
        nP_list = nP_list_total[i]
        nM_list = nM_list_total[i]
        
        #Cut off integrand that does not contribute significantly.
        elower,eupper = calc_intelligent_integrand(
          Hamiltonian0 ,
          GammaLP,GammaR,
          GammaLM,
          U,nP_list, nM_list,
          betaL,betaR,muL,muR,15)

        energies = np.linspace(elower,eupper,4000)
        
        #Points that could show divergent behavour are sought
        energies_zero_der,energies_zero_2der = HIA_error.energies_max_deltaT(energies,Hamiltonian0,
                                       GammaLP,GammaLM,
                                       GammaR,U,
                                       nP_list,nM_list,muL,muR,betaL,betaR)


        energies_carefull = jointwolist(energies_zero_der,energies_zero_2der)
        subdiv_limit = 100 + int(len(energies_carefull)*1.5)
        #Delta I and it's error are calculated
        DIV,DI_error = quad(negf_HIA_git.integrand_deltaI_HIA,elower ,eupper,
                         args= (Hamiltonian0 ,
                                      GammaLP,GammaR,
                                      GammaLM,
                                      U,nP_list, nM_list,
                                      betaL,betaR,muL,muR ) 
                            ,limit = subdiv_limit
                            ,points = energies_carefull

                     )

        dI_list_quad.append(DIV)
        dI_list_error.append(DI_error)

    return dI_list_quad,dI_list_error


def calc_barI_quad(V_list_total,
                       ef,
                        Hamiltonian0 ,
                          GammaLP,GammaR,
                          GammaLM,
                          U,nP_list_total, nM_list_total,
                          betaL,betaR):
    
    
    '''
    Input:
    - V_list_total = list of voltages (positive and negative )
    - ef = Fermi energy
    - Hamiltonian0 = hamiltonian for U = 0 
    - GammaLP,GammaLM: Gamma matrices of left for positive, negative magnetization
    - GammaR: Gamma matrices of right lead
    - U =  Coulomb interaction strength
    - nP_list_total,nM_list_total = list of electron densities for positive, negative magnetization
    - betaL, betaR = beta =1/kBT of left, right lead
    Ouput:
    - List of Delta I for voltage in calculted with the quad scipy method
    - Numerical Error on Delta I
    '''

    barI_list_quad = []
    barI_list_error = []

    for i in range(len(V_list_total)):

        V = V_list_total[i]
        print('--- V = {} ---'.format(V))
        #Bias
        muL,muR = ef + V/2, ef -V/2

        #Electron densities
        nP_list = nP_list_total[i]
        nM_list = nM_list_total[i]
        
        #Cut off integrand that does not contribute significantly.
        elower,eupper = calc_intelligent_integrand(
          Hamiltonian0 ,
          GammaLP,GammaR,
          GammaLM,
          U,nP_list, nM_list,
          betaL,betaR,muL,muR,15)

        energies = np.linspace(elower,eupper,4000)
        
        #Points that could show sharp peak behavour are sought
        energies_zero_der,energies_zero_2der = HIA_error.energies_max_barT(energies,Hamiltonian0,
                                       GammaLP,GammaLM,
                                       GammaR,U,
                                       nP_list,nM_list,muL,muR,betaL,betaR)


        energies_carefull = jointwolist(energies_zero_der,energies_zero_2der)
        subdiv_limit = 100 + int(len(energies_carefull)*1.5)
        #Delta I and it's error are calculated
        DIV,DI_error = quad(negf_HIA_git.integrand_barI_HIA,elower ,eupper,
                         args= (Hamiltonian0 ,
                                      GammaLP,GammaR,
                                      GammaLM,
                                      U,nP_list, nM_list,
                                      betaL,betaR,muL,muR ) 
                            ,limit = subdiv_limit
                            ,points = energies_carefull

                     )

        barI_list_quad.append(DIV)
        barI_list_error.append(DI_error)

    return barI_list_quad,barI_list_error




def calc_dI_trapz(npoints_int,
                V_list,ef,
              Hamiltonian0 ,
              GammaLP,GammaR,
              GammaLM,
              U,nP_tot_list, nM_tot_list,
              betaL,betaR):
    
    '''
    Input:
    npoints_int = number of points to integrate over
    - V_list = list of voltages (positive and negative )
    - ef = Fermi energy
    - Hamiltonian0 = hamiltonian for U = 0 
    - GammaLP,GammaLM: Gamma matrices of left for positive, negative magnetization
    - GammaR: Gamma matrices of right lead
    - U =  Coulomb interaction strength
    - nP_list_total,nM_list_total = list of electron densities for positive, negative magnetization
    - betaL, betaR = beta =1/kBT of left, right lead
    Output:
    dIlist = values I(m) - I(-m) for voltage V
    '''
       
    dI_list = []
    
    euplist = []
    edownlist = []
    
    for i in range(len(V_list)):
        V = V_list[i]
        muL,muR = ef +V/2,ef-V/2
        nP_list = nP_tot_list[i]
        nM_list = nM_tot_list[i]
        
        #Cut off integrand that does not contribute significantly.
        elower,eupper = calc_intelligent_integrand(
          Hamiltonian0 ,
          GammaLP,GammaR,
          GammaLM,
          U,nP_list, nM_list,
          betaL,betaR,muL,muR,15)
        
        euplist.append(eupper)
        edownlist.append(elower)

        energies = np.linspace(elower,eupper,npoints_int)

        
        
        Tbar_list = [ negf_HIA_git.integrand_deltaI_HIA(energy, 
                                 Hamiltonian0 ,
                                      GammaLP,GammaR,
                                      GammaLM,
                                      U,nP_list, nM_list,
                                      betaL,betaR,muL,muR ) for energy in energies]

        dI = np.trapz(Tbar_list,energies)
        dI_list.append(dI)
        
#         plt.plot(energies,Tbar_list)
#         plt.show()
        
#     plt.plot(V_list,euplist)
#     plt.plot(V_list,edownlist)
#     plt.show()

    return dI_list


def calc_I_trapz(npoints_int,
                V_list,ef,
              Hamiltonian0 ,
              GammaL,GammaR,
              U,n_tot_list,
              betaL,betaR):
    
    '''
    Input:
    npoints_int = number of points to integrate over
    - V_list = list of voltages (positive and negative )
    - ef = Fermi energy
    - Hamiltonian0 = hamiltonian for U = 0 
    - GammaL,GammaR: Gamma matrices left,rigth lead
    - U =  Coulomb interaction strength
    - n_list_total = list of electron densities 
    - betaL, betaR = beta =1/kBT of left, right lead
    Output:
    Ilist = list of currents for voltage V
    '''
       
    I_list = []
    
    for i in range(len(V_list)):
        V = V_list[i]
        muL,muR = ef +V/2,ef-V/2
        n_list = n_tot_list[i]
        
        #Cut off integrand that does not contribute significantly.
        elower,eupper = ef - 10, ef +10

        energies = np.linspace(elower,eupper,npoints_int)

        
        
        Tbar_list = [ negf_HIA_git.integrand_current_HIA(energy,
                                 Hamiltonian0 ,
                                  GammaL,GammaR,
                                  U,n_list, 
                                  betaL,betaR,muL,muR ) for energy in energies]

        Icur = np.trapz(Tbar_list,energies)
        I_list.append(Icur)




    return I_list










def calc_intelligent_integrand(
              Hamiltonian0 ,
              GammaLP,GammaR,
              GammaLM,
              U,nP_list, nM_list,
              betaL,betaR,muL,muR,nround):
    
    '''
    Input:
    nround = numerical order one wants to neglect
    Output:
    lower and upper bound [elower,eupper] outside which the integrand is to the numerical order given by nround.
    '''
    
    elist = [muL,muR]
    e_lower = min(elist) - 2
    
    integrand_lower = abs( np.round( negf_HIA_git.integrand_deltaI_HIA(e_lower, 
                          Hamiltonian0 ,
                          GammaLP,GammaR,
                          GammaLM,
                          U,nP_list, nM_list,
                          betaL,betaR,muL,muR ),nround) )
    
    while integrand_lower != 0:
        e_lower -= 0.1
        integrand_lower = abs( np.round( negf_HIA_git.integrand_deltaI_HIA(e_lower, 
                          Hamiltonian0 ,
                          GammaLP,GammaR,
                          GammaLM,
                          U,nP_list, nM_list,
                          betaL,betaR,muL,muR ),nround) )
        
        
        
        
    e_upper = max(elist) + 2
    
    integrand_upper = abs( np.round( negf_HIA_git.integrand_deltaI_HIA(e_upper, 
                          Hamiltonian0 ,
                          GammaLP,GammaR,
                          GammaLM,
                          U,nP_list, nM_list,
                          betaL,betaR,muL,muR ),nround) )
    
    while integrand_lower != 0:
        e_upper += 0.1
        integrand_lower = integrand_upper = abs( np.round( negf_HIA_git.integrand_deltaI_HIA(e_upper, 
                          Hamiltonian0 ,
                          GammaLP,GammaR,
                          GammaLM,
                          U,nP_list, nM_list,
                          betaL,betaR,muL,muR ),nround) )
    
    
    
    
    return e_lower,e_upper
    
    








        
def func_MR_list(y1list,y2list,xlist):
    '''Input
    y1list,y2list: lists that are a function of the parameter x in xlist
    Output
    plist = list with values: 'P = (y2-y1)/(y1 + y2)' 
    xprime_list = New x parameters. Values of x for which y1(x) + y2(x) =0 are removed (0/0 is numerically impossible)'''
    
    p_list = []
    xprime_list= []
    for i in range(len(xlist)):
        x = xlist[i]
        
        y1 = y1list[i]
        y2 = y2list[i]
        
        
        
        if x!=0:
            p_list.append(100*np.subtract(y1,y2)/(y2 + y1))
            
            xprime_list.append(x)
            
    
    return xprime_list,p_list



