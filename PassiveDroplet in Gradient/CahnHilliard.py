#!/usr/bin/env python
# coding: utf-8

# # Passive Droplet in a chemical gradient in an $\infty$ cylindrical domain using Cahn-Hilliard based Modelling

# ######################################################################################################################

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math


# In[2]:


plt.rcParams.update({'font.size': 15})


# In[3]:


import h5py

import os

os.system('clear')


# In[4]:


import traceback


# In[5]:


from scipy import optimize
from sympy import *

from scipy.stats import linregress


# ### Importing the code + PyPackage in the cluster

# In[6]:


############################## BEGIN CLUSTER STUFF ##############################

# Preserve environment variables
#$ -V

# Use python as shell
#$ -S /home/ajinkya/anaconda3/bin/python3

# Execute from current working directory
#$ -cwd

##############################
#
file = open("/home/ajinkya/PyPackage_from_Github/PyPackagePath.txt", "r") # Cluster version

############################## END CLUSTER STUFF ##############################


# ### Importing the code + PyPackage locally

# In[7]:

# file = open("/Users/ajinkyakulkarni/Desktop/Droplets-Package/PyPackagePath.txt", "r") # Local laptop version

##############################

path_location = str(file.read())

path_location = path_location.splitlines()

path_location

##############################

# Check if the paths are printed properly

for paths in path_location:

    print()
    print(paths)

print()


# In[8]:


import sys


# In[9]:


for paths in path_location:

    sys.path.append(paths)


# ### Import the packages from the PyPackage

# In[10]:


import pde

import droplets

import phasesep

import agent_based

# ######################################################################################################################

# ### Switch for DROPLET tracker or STORAGE tracker

# In[11]:


tracker = 'Droplet_Tracker'

# tracker = 'Full_Field_Tracker' # Takes muchc longer to simulate

print('Tracker is', tracker)
print()


# ### Set the version of the simulations - Cluster or Local test on the laptop

# In[12]:


file = open("VERSION.out", "r")
VERSION = file.read()


# In[13]:


if (VERSION == 'TEST'):

    print('TEST version is ON')
    print()

if (VERSION == 'CLUSTER'):

    print('CLUSTER version is ON')
    print()


# ### Read the parameters

# In[14]:


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


# In[15]:


file = open("alpha.out", "r")
ALPHA = float(file.read())

file = open("beta.out", "r")
BETA = float(file.read())


# In[16]:


file = open("C_bottom_CH.out", "r")
C_bottom_CH = float(file.read())

file = open("C_top_CH.out", "r")
C_top_CH = float(file.read())

file = open("mu_bottom_CH.out", "r")
mu_bottom_CH = float(file.read())

file = open("mu_top_CH.out", "r")
mu_top_CH = float(file.read())

############################################################

file = open("C_bottom_y_AGM.out", "r")
C_bottom_y_AGM = float(file.read())

file = open("C_top_y_AGM.out", "r")
C_top_y_AGM = float(file.read())


# In[17]:


DIFFUSION = PREFACTOR_FREE_ENERGY_DENSITY * MOBILITY

K0 = DIFFUSION/(W**2)


# In[18]:


file = open("R_critical_theory.out", "r")
R_critical_theory = float(file.read())

print('Critical radius is ', R_critical_theory)


# ### Define the cylindrical grid - for Cahn-Hilliard simulations

# In[19]:


file = open("limit_r.out", "r")
limit_r = int(file.read())

file = open("limit_z.out", "r")
limit_z = int(file.read())

support_points_r = int(2 * limit_r)

support_points_z = int(2 * limit_z)

print('Radius CH - Simulations is:', limit_r, 'and length for CH - Simulations is:', limit_z)
print()


# In[20]:

CH_grid = pde.CylindricalGrid(limit_r, [-limit_z/2, limit_z/2],
                              [support_points_r, support_points_z],
                              periodic_z = False)

# ### Check is discretization is (sufficiently) less than the interface width ~ Usually about 0.5

# In[21]:


print('Grid discretization is', CH_grid.discretization)

print()

print('Interface width is', W)

print()


# Read the scalarfield

Initial_ScalarField = pde.FieldBase.from_file("Initial_Scalarfield.hdf5")

# ## Run the actual simulation

# #### Define TMAX and timestep

# In[31]:


CH_t_max = int(np.load('T_max.npy'))

print('CH_t_max is', CH_t_max)
print()


# ### Define the trackers

# #### Tracking interval

# In[32]:


desired_timepoints = int(1e3)

tracking_interval =  int(CH_t_max/desired_timepoints) # Frequency for writing out the data

print('CH_t_max is', CH_t_max, ', Desired timepoints are', desired_timepoints,
      'and tracking interval is', tracking_interval)

# ### Droplet Tracker

# In[33]:



CH_droplet_tracks_filename = str('CH_droplet_tracks_') + str('TMAX_') + str(CH_t_max) + '.hdf5'

droplet_tracker = droplets.DropletTracker(interval = tracking_interval,
                                       refine = True,
                                       filename = CH_droplet_tracks_filename)


# ### Background Tracker

# In[34]:


CH_storage_filename = str('CH_storage_') + str('TMAX_') + str(CH_t_max) + '.hdf5'

storage = pde.FileStorage(CH_storage_filename)

concentration_field_tracker = storage.tracker(interval = tracking_interval)


# #### Euler's Method

# In[36]:


euler_dt = 5e-3 # Timestep for 0.5 discretization

print('Timestep is', euler_dt)

print()


# ### Run the simulation

# In[37]:


f = phasesep.GinzburgLandau2Components()

f.expression


# In[38]:


################ BOUNDARY CONDITIONS FOR THE Cylindrical Domain ################

# # No gradient!

# mu_bc_z = [{'type': 'derivative', 'value': 0},{'type': 'derivative', 'value': 0}]

################################################################################

# Yes Gradient!

mu_bc_z = [{'type': 'value', 'value': mu_bottom_CH},{'type': 'value', 'value': mu_top_CH}]


# In[39]:


mu_bc_r = [{'type': 'derivative', 'value': 0},{'type': 'derivative', 'value': 0}]

C_bc_z = [{'type': 'derivative', 'value': 0},{'type': 'derivative', 'value': 0}]

C_bc_r = [{'type': 'derivative', 'value': 0},{'type': 'derivative', 'value': 0}]


# In[40]:


# # Passive Droplet - No reactions

equation_to_be_solved = phasesep.CahnHilliardExtendedPDE({'free_energy': 'ginzburg-landau',
                                                         'mobility':  MOBILITY, 'kappa': KAPPA,
                                                         'bc_phi':[C_bc_r, C_bc_z],
                                                          'bc2_type': "mu",
                                                          'bc2': [mu_bc_r, mu_bc_z],
                                                          'reaction_flux': None})


# In[41]:


# result = equation_to_be_solved.solve(Initial_ScalarField, t_range = CH_t_max, dt = euler_dt,
#                                      tracker = ['progress', 'plot',
#                                                 droplet_tracker,
#                                                 concentration_field_tracker]);

result = equation_to_be_solved.solve(Initial_ScalarField, t_range = CH_t_max, dt = euler_dt,
                                     tracker = ['progress', droplet_tracker]);

print()


# ### Plot the final background field

# In[42]:


file_name_initial = 'Initial_State_T_' + str(CH_t_max) + '.png'

Initial_ScalarField.plot(title = 'Initial State', vmin = C_bottom_CH, vmax = C_top_CH,
                        filename = file_name_initial, action = 'close')

#######################################################################

file_name_final = 'Final_State_T_' + str(CH_t_max) + '.png'

result.plot(title = 'Final State', vmin = C_bottom_CH, vmax = C_top_CH, filename = file_name_final, action = 'close')


# ### Make a movie

# In[43]:


# pde.visualization.movie_scalar(result, 'CH_2D_PassiveEmulsion.mov')


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


# ### Identify global droplet recording timestamps

# In[49]:


global_droplet_timestamps = np.asarray(droplet_tracker.data.times)


# ### Isolate position and radius of the droplet

# In[50]:


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


# In[51]:


single_droplet_timestamps = np.asarray(single_droplet_timestamps)

single_droplet_x_location = np.asarray(single_droplet_x_location)

single_droplet_y_location = np.asarray(single_droplet_y_location)

single_droplet_z_location = np.asarray(single_droplet_z_location)

single_droplet_radius = np.asarray(single_droplet_radius)


# ### Plot the droplet_tracks

# In[52]:


for droplet in range(N_DROPLETS):

    plt.plot(global_droplet_timestamps, single_droplet_radius[droplet],  linewidth = 3)

########################################################################

# Plot the original tracks too, just to be sure

# global_droplet_track_list.plot(linewidth = 1, linestyle = '--', color = 'k')

########################################################################

plt.xlabel(r'$t$')

plt.ylabel(r'$\frac{R}{w}$')

########################################################################

# plt.minorticks_on()
# plt.grid(b = True, which = 'both', linewidth = 0.05)

########################################################################

plt.title(str(3) + 'D CH Passive Droplet, ' + str(N_DROPLETS) + ' Droplet')

########################################################################

plt.xlim(0, CH_t_max)

#######################################################################


image_name = 'Radius_T_' + str(CH_t_max)

plt.savefig(image_name, dpi = 400, bbox_inches = 'tight')

#######################################################################

plt.close()


# In[53]:


for droplet in range(N_DROPLETS):

#     plt.plot(global_droplet_timestamps, single_droplet_x_location[droplet], linewidth = 3,
#             label = 'x-axis')

#     plt.plot(global_droplet_timestamps, single_droplet_y_location[droplet], linewidth = 3,
#             label = 'y-axis')

    plt.plot(global_droplet_timestamps, single_droplet_z_location[droplet], linewidth = 3,
            label = 'z-axis')

########################################################################

plt.xlabel(r'$t$')

plt.ylabel('Location/w')

#########################################################

# plt.minorticks_on()
# plt.grid(b = True, which = 'both', linewidth = 0.05)

#########################################################

plt.title(str(3) + 'D CH Passive Droplet, ' + str(N_DROPLETS) + ' Droplet')

#########################################################

plt.xlim(0, CH_t_max)

# plt.ylim(-0.5, 1.1 * max(radius_of_droplets))

plt.legend()

#########################################################

image_name = 'Location_T_' + str(CH_t_max)

plt.savefig(image_name, dpi = 400, bbox_inches = 'tight')

#######################################################################

plt.close()


# ### Save all the info about radius and position

# In[54]:


# np.save('CH_single_droplet_times.npy', single_droplet_timestamps)


file_name1 = 'CH_droplet_times.npy'

np.save(file_name1, global_droplet_timestamps)

####################################################################

# np.save('CH_global_droplet_radius_original.npy', single_droplet_radius)

file_name2 = 'CH_droplet_radius.npy'

np.save(file_name2, single_droplet_radius)

####################################################################

# np.save('CH_droplet_x_location.npy', single_droplet_x_location)

# np.save('CH_droplet_y_location.npy', single_droplet_y_location)

file_name3 = 'CH_droplet_z_location.npy'

np.save(file_name3, single_droplet_z_location)

# ######################################################################################################################

# In[55]:


final_string_1 = 'rm '+ CH_droplet_tracks_filename

os.system(final_string_1)

final_string_2 = 'rm '+ CH_storage_filename

os.system(final_string_2)


# ######################################################################################################################
