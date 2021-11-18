#!/usr/bin/env python
# coding: utf-8

# # Passive/Active Emulsions in a $3D$ grid

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import os, traceback, sys, h5py


# In[2]:


import pylab as p


# In[3]:


plt.rcParams.update({'font.size': 15})

############################## BEGIN CLUSTER STUFF ##############################

# Preserve environment variables
#$ -V

# Use python as shell
#$ -S /home/ajinkya/anaconda3/bin/python3

# Execute from current working directory
#$ -cwd

# #############################
#
file = open("/home/ajinkya/PyPackage_from_Github/PyPackagePath.txt", "r") # Cluster version

path_location = str(file.read())
path_location = path_location.splitlines()

import sys
for paths in path_location:
    sys.path.append(paths)

import pde, droplets, phasesep, agent_based

############################# END CLUSTER STUFF ##############################

# ### Importing the code + PyPackage locally

# In[4]:

#
# sys.path.append('/Users/ajinkyakulkarni/Desktop/GitHub/py-pde')
# import pde
#
# sys.path.append('/Users/ajinkyakulkarni/Desktop/GitHub/py-droplets')
# import droplets
#
# sys.path.append('/Users/ajinkyakulkarni/Desktop/GitHub/py-phasesep')
# import phasesep
#
# sys.path.append('/Users/ajinkyakulkarni/Desktop/GitHub/agent-based-emulsions')
# import agent_based


# ######################################################################################################################

# In[5]:


import os

os.system('rm *.txt *.hdf5 *.png *.npy')


# ### Define number of drops and distribution

# In[ ]:


N_DROPLETS = int(1e5)


# In[ ]:


MEAN_RADIUS = 10


# In[ ]:


radius_of_droplets = np.random.uniform(0.95*MEAN_RADIUS, 1.05*MEAN_RADIUS, N_DROPLETS)


# ### Define AGM SYSTEM SIZE

# In[ ]:


SYSTEM_SIZE = int(1e3 * MEAN_RADIUS)

print('SYSTEM_SIZE is', SYSTEM_SIZE)

print()


# ### Define position of drops

# In[ ]:


AGM_grid = pde.CartesianGrid([(0, SYSTEM_SIZE)] * 3, 1, periodic=True)


# ######################################################################################################################

# In[ ]:


position_of_droplets = []

for i in range(N_DROPLETS):

    position_of_droplets.append(AGM_grid.get_random_point())

position_of_droplets = np.asarray(position_of_droplets)


# In[ ]:


list_of_droplets = [droplets.SphericalDroplet(position = position_of_droplets[i], radius = radius_of_droplets[i])

                    for i in range(N_DROPLETS)]


# In[ ]:


Initial_Emulsion = droplets.Emulsion(droplets = list_of_droplets, grid = AGM_grid)


# ### Calculate and save mean surface_separation

# In[ ]:


mean_droplet_surface_separation = int(np.mean(Initial_Emulsion.get_neighbor_distances(subtract_radius = True)))


# #####################################################################################################################

# In[ ]:


background = agent_based.DiffusionBackground()


# In[ ]:


initial_background = pde.ScalarField(AGM_grid, 0.05)


# In[ ]:


agents = agent_based.SphericalDropletAgents({'equilibrium_concentration': str(2*0.083) + str('/radius')})


# In[ ]:


simulation = agent_based.AgentSimulation(background, agents)


# In[ ]:


background_plus_agents = simulation.get_state(background = initial_background, agents = list_of_droplets)


# In[ ]:


AGM_t_max = int(1e10)

# AGM_t_max = int(1e5)

#########################

desired_timepoints = int(1e6)

# desired_timepoints = int(1e3)

#########################

tracking_interval = int(AGM_t_max/desired_timepoints) # Frequency for writing out the data


# In[ ]:


droplet_tracker = agent_based.DropletAgentTracker(interval = tracking_interval,
                                                  store_droplet_tracks = True,
                                                  store_emulsions = True)


# ### Run the simulation

# In[ ]:


timestep = int(1e3)*simulation.estimate_dt(background_plus_agents)


# In[ ]:


result = simulation.run(background_plus_agents, t_range = AGM_t_max,
                        dt = timestep,
                        tracker = ['progress', droplet_tracker],
                        backend = 'numba')

print()


# ### Save all data

# In[ ]:


global_droplet_timestamps = np.asarray(droplet_tracker.emulsions.times)

np.save('global_droplet_timestamps.npy', global_droplet_timestamps)


# In[ ]:


global_droplet_number = []

for snapshot in range(len(global_droplet_timestamps)):

    global_droplet_number.append(len(droplet_tracker.emulsions.emulsions[snapshot]))

np.save('global_droplet_number.npy', global_droplet_number)


# In[ ]:


array = []

for snapshot in range(len(global_droplet_timestamps)):

    array.append([global_droplet_timestamps[snapshot],
                 len(droplet_tracker.emulsions.emulsions[snapshot]),
                 droplet_tracker.emulsions.emulsions[snapshot].get_size_statistics()['radius_mean'],
                 (droplet_tracker.emulsions.emulsions[snapshot].get_linked_data()['radius'])\
                 /(droplet_tracker.emulsions.emulsions[snapshot].get_size_statistics()['radius_mean'])])

array = np.asarray(array)

np.save('global_array.npy', array)


# ######################################################################################################################
