#Author: Karssien Hero Huisman

#Notebook for functions that appear throughout both the Hartree-Fock and Hubbard-One notebooks.
import numpy as np

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


def check_listzero(alist,tol):

    '''
    Input
    - two lists: lista,listb
    - tol = convergence criterium
    Ouput:
    check_zero = absolute value of the difference between lista,listb rounded by tol
    zero_bool = True if convergence is achieved'''
    acc = int(np.ceil(-np.log(tol)/np.log(10))) 
    check_zero = np.round(alist,acc)

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

def check_difference2(lista,listb,tol):

    '''
    Input
    - two lists: lista,listb
    - tol = convergence criterium
    Ouput:
    check_zero_prune = absolute value of the difference between lista,listb subtracted by tol
    zero_bool = True if convergence is achieved'''
    check_zero = abs(np.subtract(lista,listb))


    dim = len(check_zero)
    teller = 0

    check_zeroprime = []
    for element in check_zero:


        if element <= tol:
            teller +=  1
            check_zeroprime.append(0)
        if element > tol:
            check_zeroprime.append(element)
        

    if teller == dim:
        zero_bool = True

    if teller != dim:
        zero_bool = False
        
    return check_zeroprime, zero_bool


def halves_list(Hamiltonian):
    
    ''' Input:
    - N X N Hamiltonian
    Output:
    - list: = [1/2,1/2,....] length of the list is N.
    '''
    
    number_sites_spin,number_sites_spin = Hamiltonian.shape

    n_list_s = [0.5   for i in range(number_sites_spin)]
    
    return n_list_s

def list_halves(Hamiltonian):
    
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

def func_V_total(Vmax,dV):
    
    '''
    Input: 
    Vmax = maximum bias voltage
    dV   = voltage stepsize
    Output: 
    V_list_pos_bias : list with only positive voltages
    V_list_total    : list off all voltages
    '''
    
    V_list_pos_bias = np.arange(dV,Vmax + dV,dV)

    V_list_neg_bias = -1*np.flip(V_list_pos_bias)

    V_list_total = jointwolist(jointwolist(V_list_neg_bias,[0]),V_list_pos_bias)

    return V_list_pos_bias,V_list_total

def check_list_smaller(alist,value):

    '''
    Input
    - alist = list with values
    - value = positive number
    Ouput:
    zero_bool = True if values in alist are smaller than 'value' and bigger or equal to zero:
                0 <= alist[i] <= value '''

    dim = len(alist)
    teller = 0


    for element in alist:


        if element <= value and element > 0:
            teller +=  1

    if teller == dim:
        zero_bool = True

    if teller != dim:
        zero_bool = False
        
    return zero_bool
        

   













