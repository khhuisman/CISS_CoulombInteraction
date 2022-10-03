
# Notebook to calculate magnetocurrent in Hartree Fock Approximation
from scipy.integrate import quad
from matplotlib import pyplot as plt
import numpy as np


# #### Non - Equilibrium Green's Function method


import negf_git 

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
    number_sites_spin,number_sites_spin = Hamiltonian.shape
    
    ilist = [i   for i in range(number_sites_spin)]
    
    return ilist


def calc_intelligent_integrand(HP,HM,
                                 GammaLP,GammaLM,
                                 GammaR,
                              betaL,betaR,muL,muR,nround):
    
    '''
    Input:
    nround = numerical order one wants to neglect
    Output:
    lower and upper bound [elower,eupper] outside which the integrand is to the numerical order given by nround.
    '''
    
    elist = [muL,muR]
    e_lower = min(elist) - 1
    
    integrand_lower = abs( np.round( negf_git.integrand_current_deltaI(e_lower, 
                                                                         HP,HM,
                                                                         GammaLP,GammaLM,
                                                                         GammaR,
                                                                      betaL,betaR,muL,muR
                                                                  ),nround
                                   )
                         ) 
    
    while integrand_lower != 0:
        e_lower -= 0.1
        integrand_lower = abs( np.round( negf_git.integrand_deltaI_HIA(e_lower, 
                          Hamiltonian0 ,
                          GammaLP,GammaR,
                          GammaLM,
                          U,nP_list, nM_list,
                          betaL,betaR,muL,muR ),nround) 
                             )
        
        
        
        
    e_upper = max(elist) + 1
    
    integrand_upper = abs( np.round( negf_git.integrand_current_deltaI(e_upper, 
                                 HP,HM,
                                 GammaLP,GammaLM,
                                 GammaR,
                              betaL,betaR,muL,muR),nround) )
    
    
    while integrand_lower != 0:
        e_upper += 0.1
        integrand_lower = integrand_upper = abs( np.round( negf_git.integrand_deltaI_HIA(e_upper, 
                          Hamiltonian0 ,
                          GammaLP,GammaR,
                          GammaLM,
                          U,nP_list, nM_list,
                          betaL,betaR,muL,muR ),nround) )
    
    
    
    
    return e_lower,e_upper

##### Calculate current with trapezoid method ######



# In[ ]:



def calc_dI_trapz(npoints_int,
                V_list,ef,
              Hamiltonian0 ,
              GammaLP,GammaR,
              GammaLM,
              U,nP_tot_list, nM_tot_list,
              betaL,betaR):
    
    '''
    Output:
    dIV = values I(m) - I(-m) for voltage V
    dIV_error = error bound for the trapezoid method.
    energies = energies for which to calculate dIV
    Tbar_list = Integrand as function of energy.
    '''
       
    dI_list = []
    
    for i in range(len(V_list)):
        V = V_list[i]
        muL,muR = ef +V/2,ef-V/2
        nP_list = nP_tot_list[i]
        nM_list = nM_tot_list[i]
        
        HP = negf_git.Hamiltonian_HF(nP_list,U,Hamiltonian0)
        HM = negf_git.Hamiltonian_HF(nM_list,U,Hamiltonian0)
                          
                          
        elower,eupper = calc_intelligent_integrand(HP,HM,
                                 GammaLP,GammaLM,
                                 GammaR,
                              betaL,betaR,muL,muR,15)
                          
#         elower_list.append(elower) 
#         eupper_list.append(eupper) 
        
        energies = np.linspace(elower,eupper,npoints_int)
        
        
        Tbar_list = [ negf_git.integrand_current_deltaI(energy, 
                                 HP,HM,
                                 GammaLP,GammaLM,
                                 GammaR,
                              betaL,betaR,muL,muR) for energy in energies]

        dI = np.trapz(Tbar_list,energies)
        dI_list.append(dI)


    


    return dI_list


import negf_deltaT_git as negf_DT


def calc_dI_quad(
                V_list,ef,
              Hamiltonian0 ,
              GammaLP,GammaR,
              GammaLM,
              U,nP_tot_list, nM_tot_list,
              betaL,betaR):
    
    '''
    Output:
    - Magnetocurrent: DeltaI(m,V) = I(m,V)-I(-m,V) as a function of bias voltage
    - Estimate of numerical error as given by quad
    '''
    
    
    
    dI_list = []
    dI_list_err = []
                          
    elower_list = []
    eupper_list = []

    
    
    for i in range(len(V_list)):
        V = V_list[i]
        
        print( ''' --- V = {} ---'''.format(V))
        muL,muR = ef +V/2,ef-V/2
        nP_list = nP_tot_list[i]
        nM_list = nM_tot_list[i]
        
        HP = negf_git.Hamiltonian_HF(nP_list,U,Hamiltonian0)
        HM = negf_git.Hamiltonian_HF(nM_list,U,Hamiltonian0)
                          
                          
        elower,eupper = calc_intelligent_integrand(HP,HM,
                                 GammaLP,GammaLM,
                                 GammaR,
                              betaL,betaR,muL,muR,15)
                          
        
        energies = np.linspace(elower,eupper,300)
        dE_DTlist =  [ negf_DT.DE2_DeltaTbar(energy,HP,HM,GammaLP,GammaLM,GammaR,muL,muR,betaL,betaL)  for energy in energies]
        energies_zero_der1 = negf_DT.func_deriv_zero(energies,np.array(dE_DTlist)[:,1])
        energies_zero_der2 = negf_DT.func_deriv_zero(energies,np.array(dE_DTlist)[:,2])
        
        energies_carefull = jointwolist(energies_zero_der1,energies_zero_der2)
        subdiv_limit = 100 + int(len(energies_carefull)*1.5)
        
        
        
        dI,dIerror = quad(negf_git.integrand_current_deltaI, elower-10,eupper+10, args = ( 
                                 HP,HM,
                                 GammaLP,GammaLM,
                                 GammaR,
                              betaL,betaR,muL,muR) ,
                              limit = subdiv_limit,
                              points = energies_carefull
                        )

        dI_list.append(dI)
        dI_list_err.append(dIerror)
                          
                          
                          
    

    return dI_list,dI_list_err


    




