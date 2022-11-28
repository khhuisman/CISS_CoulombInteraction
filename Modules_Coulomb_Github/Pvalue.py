#!/usr/bin/env python
# coding: utf-8

#Author: Karssien Hero Huisman


import numpy as np




def evenodd_index_list(xlist):
    '''
    Input:  A list with positive and negative values.
    Output: Indices of the list that are anti-symmetric in flipping the list:
        ex: In: [-3,-2.5,-1,0,1,2,3] 
                        ->  
                [-3,-1,0,1,3] is the anti-symmetric part in flipping.
            Out: [0,2,3,4,6]
    '''
    index_sol_list = []

    for i in range(len(xlist)):
        
        x = xlist[-1-i]
        if x !=0:
            sol_x = np.where(np.round(xlist + x,11) == 0)[0]
            xrounded = np.round(x,12)
            if len(sol_x) == 1:
                index_sol_list.append(sol_x[0])
                
        if x == 0:
            index_sol_list.append(i)

    return index_sol_list

def P_list_trapz(Odd_list_squared,Even_list_squared,xlist):
    
    '''
    Input:
    Odd_list_squared = odd values squared
    Even_list = even values squared
    xlist = Variable in which input lists are odd or even.
    Output:
    P values as a function of x
    '''

    n = int(len(xlist)/2)
    xlist_prime = []
    PJ_list = []
    for i in range(1,n+1):
        slices = n-i

        if slices !=0:
            xlist_int = xlist[slices:-slices]
            Even_list_int = Even_list_squared[slices:-slices]
            Odd_list_int = Odd_list_squared[slices:-slices]
        if slices == 0:
            xlist_int = xlist
            Even_list_int = Even_list_squared
            Odd_list_int = Odd_list_squared
        
    
        Peven = np.trapz(Even_list_int,xlist_int)
        Podd = np.trapz(Odd_list_int,xlist_int)
        
        if Peven+Podd !=0:
            PJ_list.append((Podd-Peven)/(Peven+Podd))
            xlist_prime.append(max(xlist_int))
        
        
    return xlist_prime,PJ_list

def function_PvaluedI(V_list,dI_list,int_acc):
    
    '''
    Input:
    - Vlist = list of voltages
    - dI_list = list of magnetocurrent
    - int_acc = accuracy of numerical integration (we round "int_acc" number behind the comma)
    Output:
    - Plist = list with P values
    - V_list_prime = list which is anti-symmetric in flipping all elements (with np.flip())
    '''
    
    #Selection on voltages s.t. Delta I can be decomposed in odd,even part.
    index_sol_list = evenodd_index_list(V_list) 
    V_list_correct  = [V_list[k] for k in index_sol_list]
    dI_list = np.round([dI_list[k] for k in index_sol_list],int_acc)
    dIbar_list = np.flip(dI_list) # since leads are symmtrically coupled we can simply flip

    #Odd,Even part of Delta I
    Even_dIlist = np.multiply(0.5,np.add(dI_list,dIbar_list))
    Odd_dIlist = np.multiply(0.5,np.subtract(dI_list,dIbar_list))

    
    Even_list_squared = [Even_dIlist[i]**2 for i in range(len(Even_dIlist))]
    Odd_list_squared  = [Odd_dIlist[i]**2 for i in range(len(Odd_dIlist))]
    
    Vlist_prime,P_list = P_list_trapz(Odd_list_squared,Even_list_squared,V_list_correct)
    
    return Vlist_prime,P_list

