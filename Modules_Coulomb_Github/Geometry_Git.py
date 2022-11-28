
#Author: Karssien Hero Huisman

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#### Geometrical Parameters of SOC

def rvec(m,a,c,M,N,chirality):
    
    '''
    Input:
    - N = number of sites per lap ( 1 lap = rotation by 2pi around helix axis)
    - M = number of laps
    - c = length of the molecule
    - a = radius
    - m = site label
    - chirality = boolean indicating the chirality of the helix.
    Ouput:
    - positional vector of helix
    
    '''
    if chirality==True:
        sign = 1
    if chirality== False:
        sign = -1
        
    phim = 2*np.pi*(m-1)/N # Fransson
#     phi0 = np.pi/2
    Rpos = [a*np.cos(phim),a*np.sin(sign*phim),c*(m-1)/(M*N-1)] # Fransson


    return Rpos

def dvec(m,stilde,a,c,M,N,chirality):
    
    '''
    Input:
    - N = number of sites per lap ( 1 lap = rotation by 2pi around helix axis)
    - M = Number of laps
    - c = length of the molecule
    - a = radius
    - m = site label
    - stilde = integer, for a site 1,2,3,.. next to site m 
    - chirality = boolean indicating the chirality of the helix.
    Ouput:
    - vector D(m+s): difference between positional vector:  on site m and m+s.
    '''
    
    Rm = rvec(m,a,c,M,N,chirality)
    Rms = rvec(m+stilde,a,c,M,N,chirality)
    
    dRmRms = np.subtract(Rm,Rms)
    norm = np.linalg.norm(dRmRms)
    
    Dvector = np.multiply(1/norm,dRmRms)
    
    return Dvector


def Vmvec(m,stilde,a,c,M,N,chirality=True):
    
    '''
    Input:
    - N = number of sites per lap ( 1 lap = rotation by 2pi around helix axis)
    - M = Number of laps
    - c = length of the molecule
    - a = radius
    - m = site label
    - stilde = integer, for a site 1,2,3,.. next to site m
    - chirality = boolean indicating the chirality of the helix.
    Ouput:
    -  cross procducht between D(m+s) and D(m+2s).
    '''
    
    dms = dvec(m,stilde,a,c,M,N,chirality)
    dm2s = dvec(m,2*stilde,a,c,M,N,chirality) # FRANSSON
    
    vvec = np.cross(dms,dm2s)
    
    return vvec

def plot3d_chain(a,c,M,N):
    
    '''
    Input:
    - N = number of sites per lap ( 1 lap = rotation by 2pi around helix axis)
    - M = Number of laps
    - c = length of the molecule
    - a = radius
    - chirality = boolean indicating the chirality of the helix.
    Ouput:
    - 3D scatter plot of the chain
    '''
    m_max = M*N
    
    chirality_list =[True,False]
    positions_list = []
    
    for chirality in chirality_list:
        xlist = []
        ylist = []
        zlist = []
        for i in range(1,m_max +1):
            x,y,z = rvec(i,a,c,M,N,chirality)
            xlist.append(x)
            ylist.append(y)
            zlist.append(z)
        positions_list.append([xlist,ylist,zlist])
        
        
        
     
    for i in range(len(chirality_list)):
        chirality = chirality_list[i]
        xlist,ylist,zlist = positions_list[i]
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        if chirality == True:
            ax.scatter(xlist, ylist,zlist, label='Chirality True',c ='blue')
        if chirality == False:
            ax.scatter(xlist, ylist,zlist, label='Chirality False',c ='red')
   
        ax.legend()
        plt.show() 


        
        


