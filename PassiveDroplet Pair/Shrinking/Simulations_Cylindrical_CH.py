#!/usr/bin/env python
# coding: utf-8

# ## Droplet pair

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math


# In[2]:


plt.rcParams.update({'font.size': 15})

import traceback


# ### Importing the code + PyPackage in the cluster

# In[5]:


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

# import sys
# sys.path.append('/Users/ajinkyakulkarni/Desktop/GitHub/py-pde')
# sys.path.append('/Users/ajinkyakulkarni/Desktop/GitHub/py-droplets')
# sys.path.append('/Users/ajinkyakulkarni/Desktop/GitHub/py-phasesep')
# sys.path.append('/Users/ajinkyakulkarni/Desktop/GitHub/agent-based-emulsions')
#
# import pde, droplets, phasesep, agent_based

 ######################################################################################################################

# In[10]:


import h5py


# In[11]:


file = open("KAPPA.out", "r")
KAPPA = float(file.read())

file = open("GAMMA.out", "r")
GAMMA = float(file.read())

file = open("INTERFACE_WIDTH.out", "r")
W = float(file.read())

file = open("PREFACTOR_FREE_ENERGY_DENSITY.out", "r")
PREFACTOR_FREE_ENERGY_DENSITY = float(file.read())

# This is value of b in f = (b/2)*c*c*(1-c)*(1-c)

file = open("MOBILITY.out", "r")
MOBILITY = float(file.read())


# In[12]:


file = open("limit_r.out", "r")
limit_r = int(file.read())

file = open("limit_z.out", "r")
limit_z = int(file.read())

file = open("AGM_box_length.out", "r")
AGM_box_length = int(file.read())


# In[13]:


DIFFUSION = PREFACTOR_FREE_ENERGY_DENSITY*MOBILITY

K0 = DIFFUSION/(W**2)


# In[14]:


file = open("C_INF.out", "r")
C_INF = float(file.read())

print('C_INF is', C_INF)
print()


# In[15]:


if (C_INF > 0):

    file = open("R_critical_theory.out", "r")
    R_critical_theory = float(file.read())

    print('Critical radius is ', R_critical_theory)
    print()


# ######################################################################################################################

# In[16]:


support_points_CH = int(2 * limit_r)

print('Radius CH - Simulations is:', limit_r, 'and length for CH - Simulations is:', limit_z)
print()
print('Support_points are', support_points_CH)
print()


# In[17]:


CH_grid = pde.CylindricalGrid(limit_r, [-limit_z/2, limit_z/2], [support_points_CH, support_points_CH],
                              periodic_z = False)


# ### Check is discretization is (sufficiently) less than the interface width ~ Usually about 0.5

# In[18]:


print('Grid discretization is', CH_grid.discretization)

print()

print('Interface width is', W)

print()


# ### Make a droplet pair

# In[19]:


initial_radius_list_CH = np.load('initial_radius_list_CH.npy')

print('initial_radius_list_CH is', initial_radius_list_CH)

print()


# ### Separation distance between the droplet pair

# In[20]:


initial_separation = int(np.load('initial_separation.npy'))

print('Initial_separation is', initial_separation)

print()


# In[21]:


droplet_1 = droplets.DiffuseDroplet([0, 0, -initial_separation/2], initial_radius_list_CH[0], W)

field_1 = droplet_1.get_phase_field(CH_grid)

droplet_2 = droplets.DiffuseDroplet([0, 0, initial_separation/2], initial_radius_list_CH[1], W)

field_2 = droplet_2.get_phase_field(CH_grid)


# In[22]:


droplet_pair = (field_1 + field_2)

# droplet_pair.plot('image', vmin = 0, vmax = 1, colorbar = True)


# ### Convert total_field into a ScalarField

# In[23]:


field_droplet_pair = pde.ScalarField(CH_grid, droplet_pair)

supersaturation =  pde.ScalarField(CH_grid, C_INF)

total_field_ScalarField = field_droplet_pair + supersaturation


# In[24]:


total_field_ScalarField.data = np.where(total_field_ScalarField.data > 1,
                                        (1 + 2*GAMMA/initial_radius_list_CH[0]),
                                        total_field_ScalarField.data)


# ### Calculate the deviation of the droplet radius from smoothening vs original intended radius

# In[25]:


total_field_ScalarField = total_field_ScalarField.smooth()


# In[26]:


initial_emulsion = droplets.locate_droplets(total_field_ScalarField, refine = True)

initial_emulsion.get_size_statistics()


# In[27]:


detected_intial_radius = initial_emulsion.get_size_statistics()['radius_mean']


# In[28]:


error_detected_intial_radius = (100 * abs(detected_intial_radius -
                                          initial_radius_list_CH[0]))/initial_radius_list_CH[0]


# In[29]:


print('Error in detected radius is', round(error_detected_intial_radius, 2), '%')

print()

################################################################################

if (error_detected_intial_radius > 1):

    raise ValueError('Large error detected. Decrease level of smoothening?')


# ### Save this radius as the input for AGM simulations

# In[30]:


initial_radius_list_AGM = [initial_emulsion.get_size_statistics()['radius_mean'],
                           initial_emulsion.get_size_statistics()['radius_mean']]

initial_radius_list_AGM


# In[31]:


np.save('initial_radius_list_AGM.npy', initial_radius_list_AGM)


# ## Run the actual simulation

# #### Define TMAX and timestep

# In[32]:


# CH_t_max = int(np.load('T_max.npy'))

CH_t_max = int(6e3)

print('CH_t_max is', CH_t_max)
print()


# ### Define the trackers

# #### Tracking interval

# In[33]:


desired_timepoints = int(1e2)


# In[34]:


tracking_interval =  int(CH_t_max/desired_timepoints) # Frequency for writing out the data

print('CH_t_max is', CH_t_max, ', Desired timepoints are', desired_timepoints,
      'and tracking interval is', tracking_interval)


# ### Droplet Tracker

# In[35]:


droplet_tracker = droplets.DropletTracker(interval = tracking_interval,
                                       refine = True,
                                       filename = 'CH_droplet_tracks.hdf5')


# ### Background Tracker

# In[36]:


# storage = pde.FileStorage('CH_concentration_field.hdf5')


# In[37]:


# concentration_field_tracker = storage.tracker(interval = tracking_interval)


# #### Euler's Method

# In[38]:


euler_dt = 5e-3 # Timestep for 0.5 discretization

print('Timestep is', euler_dt)

print()


# ### Run the simulation

# In[39]:


f = phasesep.GinzburgLandau2Components()

f.expression


# In[40]:


mu_bc_r = [{'type': 'derivative', 'value': 0},{'type': 'derivative', 'value': 0}]

mu_bc_z = [{'type': 'derivative', 'value': 0},{'type': 'derivative', 'value': 0}]

C_bc_z = [{'type': 'derivative', 'value': 0},{'type': 'derivative', 'value': 0}]

C_bc_r = [{'type': 'derivative', 'value': 0},{'type': 'derivative', 'value': 0}]


# In[41]:


# # Passive Droplet - No reactions

equation_to_be_solved = phasesep.CahnHilliardExtendedPDE({'free_energy': 'ginzburg-landau',
                                                         'mobility':  MOBILITY, 'kappa': KAPPA,
                                                         'bc_phi':[C_bc_r, C_bc_z],
                                                          'bc2_type': "mu",
                                                          'bc2': [mu_bc_r, mu_bc_z],
                                                          'reaction_flux': None})


# In[42]:


# result = equation_to_be_solved.solve(total_field, t_range = CH_t_max, dt = euler_dt,
#                                      tracker = ['progress', 'plot',
#                                                 droplet_tracker,
#                                                 concentration_field_tracker]);

result = equation_to_be_solved.solve(total_field_ScalarField,
                                     t_range = CH_t_max,
                                     dt = euler_dt,
                                     tracker = ['progress',
                                                droplet_tracker]);

print()


# ### Plot the final background field

# In[43]:


# total_field_ScalarField.plot(title = 'Initial State', vmin = 0, vmax = 1)

# result.plot(title = 'Final State', vmin = 0, vmax = 1)


# ### Make an Emulsion just from the final state

# In[44]:


final_emulsion = droplets.locate_droplets(result, refine = True)

final_emulsion.get_size_statistics()


# ### Check the analytical interface width and the simulation interface width

# In[45]:


print('Analytical interface width is:', final_emulsion.interface_width,
      'and Simulation interface width is:', W)

print()


# In[46]:


relative_tolerance = 1e-1

if np.isclose(final_emulsion.interface_width, W, rtol = relative_tolerance) == True:

    print('Analytical and simulation interface widths are the same within a tolerance of', relative_tolerance)
    print()

else:

    raise ValueError('Decrease relative_tolerance')


# ### Create a droplet track list for each droplet

# In[47]:


global_droplet_track_list = droplets.DropletTrackList.from_emulsion_time_course(droplet_tracker.data)


# ### Plot droplet tracks

# In[48]:


N_DROPLETS = final_emulsion.get_size_statistics()['count']

print('N_DROPLETS is', N_DROPLETS)

print()


# ### Identify global droplet recording timestamps

# In[49]:


global_droplet_timestamps = np.asarray(droplet_tracker.data.times)


# ### Isolate position and radius of the droplet

# ##### Check that len(global_droplet_track_list) is same as N_DROPLETS

# In[50]:


if (len(global_droplet_track_list) != N_DROPLETS):

    raise ValueError('Check the droplet track list')


# In[51]:


single_droplet_timestamps = []

single_droplet_x_location = []

single_droplet_y_location = []

single_droplet_z_location = []

single_droplet_radius = []

for droplet in range(N_DROPLETS):

    single_droplet_track = global_droplet_track_list[droplet]

    single_droplet_track_info = single_droplet_track.data

    ########################################################################

    single_droplet_timestamps.append(single_droplet_track_info['time'])

    single_droplet_x_location.append(single_droplet_track_info['position'][:, 0])

    single_droplet_y_location.append(single_droplet_track_info['position'][:, 1])

    single_droplet_z_location.append(single_droplet_track_info['position'][:, 2])

    single_droplet_radius.append(single_droplet_track_info['radius'])

    ########################################################################


# In[52]:


single_droplet_timestamps = np.asarray(single_droplet_timestamps)

single_droplet_x_location = np.asarray(single_droplet_x_location)

single_droplet_y_location = np.asarray(single_droplet_y_location)

single_droplet_z_location = np.asarray(single_droplet_z_location)

single_droplet_radius = np.asarray(single_droplet_radius)


# ### Save all the info about radius and position

# In[53]:


# np.save('CH_single_droplet_times.npy', single_droplet_timestamps)

np.save('CH_droplet_times.npy', global_droplet_timestamps)

####################################################################

# np.save('CH_global_droplet_radius_original.npy', single_droplet_radius)

np.save('CH_droplet_radius.npy', single_droplet_radius)

####################################################################

np.save('CH_droplet_x_location.npy', single_droplet_x_location)

np.save('CH_droplet_y_location.npy', single_droplet_y_location)

np.save('CH_droplet_z_location.npy', single_droplet_z_location)


# ### Droplet mean radius vs time

# In[54]:


CH_MeanRadius = (single_droplet_radius[0] + single_droplet_radius[1])/2


# In[55]:


plt.plot(global_droplet_timestamps, CH_MeanRadius, linewidth = 3)

# plt.minorticks_on()
# plt.grid(b=True, which='both', linewidth=0.05)

plt.title('CH-Mean radius vs Time')

plt.xlabel(r'$t$')
plt.ylabel(r'$\left \langle R \right \rangle$')

plt.savefig('CH_MeanRadius.png', dpi = 400, bbox_inches = 'tight')

plt.close()


# In[56]:


np.save('CH_MeanRadius.npy', CH_MeanRadius)


# ### Droplet separation vs time

# In[57]:


separation_distance = np.sqrt((single_droplet_x_location[0] - single_droplet_x_location[1])**2 +
                              (single_droplet_y_location[0] - single_droplet_y_location[1])**2 +
                              (single_droplet_z_location[0] - single_droplet_z_location[1])**2)


# In[58]:


# plt.plot(global_droplet_timestamps, separation_distance)

# # plt.minorticks_on()
# # plt.grid(b=True, which='both', linewidth=0.05)

# plt.title('CH-Droplet separation vs Time')

# plt.xlabel(r'$t$')
# plt.ylabel('Droplet separation')

# plt.savefig('CH_SD.png', dpi = 400, bbox_inches = 'tight')

# plt.close()


# In[59]:


np.save('CH_SeparationDistance.npy', separation_distance)


# ######################################################################################################################

# In[60]:


import os

os.system('rm CH_droplet_tracks.hdf5')

# os.system('rm CH_concentration_field.hdf5')


# ######################################################################################################################

# In[ ]:
