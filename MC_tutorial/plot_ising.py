########## Machine Learning for Quantum Matter and Technology  ######################
### Juan Carrasquilla, Estelle Inack, Giacomo Torlai, Roger Melko
### with code from Lauren Hayward Sierens/PSI
### Tutorial 1: Monte Carlo for the Ising model
#####################################################################################

import matplotlib.pyplot as plt
import numpy as np

### Input parameters (these should be the same as in ising_mc.py): ###
T_list = np.linspace(5.0,0.5,10) #temperature list
L = 4                            #linear size of the lattice
N_spins = L**2                   #total number of spins
J = 1                            #coupling parameter

### Critical temperature: ###
Tc = 2.0/np.log(1.0 + np.sqrt(2))*J

### Observables to plot as a function of temperature: ###
energy   = np.zeros(len(T_list))
mag      = np.zeros(len(T_list))
specHeat = np.zeros(len(T_list))
susc     = np.zeros(len(T_list))

### Loop to read in data for each temperature: ###
for (iT,T) in enumerate(T_list):
  file = open('Data/ising2d_L%d_T%.4f.txt' %(L,T), 'r')
  data = np.loadtxt( file )

  E   = data[:,1]
  M   = abs(data[:,2])

  energy[iT] = np.mean(E)
  mag[iT]    = np.mean(M)
  
  # *********************************************************************** #
  # *********** 2b) FILL IN CODE TO CALCULATE THE SPECIFIC HEAT *********** #
  # ***********               AND SUSCEPTIBILITY                *********** #
  # *********************************************************************** #
  specHeat[iT] = 0
  susc[iT]     = 0
#end loop over T

plt.figure(figsize=(8,6))

plt.subplot(221)
plt.axvline(x=Tc, color='k', linestyle='--')
plt.plot(T_list, energy/(1.0*N_spins), 'o-')
plt.xlabel('$T$')
plt.ylabel('$<E>/N$')

plt.subplot(222)
plt.axvline(x=Tc, color='k', linestyle='--')
plt.plot(T_list, mag/(1.0*N_spins), 'o-')
plt.xlabel('$T$')
plt.ylabel('$<|M|>/N$')

plt.subplot(223)
plt.axvline(x=Tc, color='k', linestyle='--')
plt.plot(T_list, specHeat/(1.0*N_spins), 'o-')
plt.xlabel('$T$')
plt.ylabel('$C_V/N$')

plt.subplot(224)
plt.axvline(x=Tc, color='k', linestyle='--')
plt.plot(T_list, susc/(1.0*N_spins), 'o-')
plt.xlabel('$T$')
plt.ylabel('$\chi/N$')

plt.suptitle('%d x %d Ising model' %(L,L))

plt.show()
