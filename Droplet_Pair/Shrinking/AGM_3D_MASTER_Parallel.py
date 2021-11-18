#!/usr/bin/env python
# coding: utf-8

# # Droplet Pair in an $\infty ~3D$ domain using Agent based Modelling using a Cartesian grid

# ######################################################################################################################

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math

import os

# In[2]:


plt.rcParams.update({'font.size': 10})


# In[4]:


import traceback


# ### Importing the code + PyPackage in the cluster

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

# ######################################################################################################################

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

# print('C_INF is', C_INF)
# print()


# In[15]:

if (C_INF > 0):

    file = open("R_critical_theory.out", "r")
    R_critical_theory = float(file.read())

    print('Critical radius is ', R_critical_theory)
    print()


# ######################################################################################################################

# ### Define the 3D Cartesian grid - for Agent based Model simulations

# In[16]:

# number_support_points_GRID = int(AGM_box_length/CHANGE_grid_discretization)

##################################################################

# if ((number_support_points_GRID % 2) != 0):
#
#     number_support_points_GRID = number_support_points_GRID + 1



# In[17]:


# print('Support_points for AGM grid are', number_support_points_GRID)
# print()
# print('System size for AGM is ', AGM_box_length)
# print()
#

# In[18]:


# np.save('number_support_points_GRID.npy', number_support_points_GRID)


# In[19]:


AGM_grid = pde.CartesianGrid([[-AGM_box_length/2, AGM_box_length/2],
                              [-AGM_box_length/2, AGM_box_length/2],
                              [-AGM_box_length/2, AGM_box_length/2]],
                             [int(CHANGE_support_points_GRID), int(CHANGE_support_points_GRID), int(CHANGE_support_points_GRID)],
                             periodic = [False, False, False])


# In[20]:


# print('AGM_grid_discretization is', AGM_grid.discretization)
#
# print()
#
# np.save('AGM_grid_discretization.npy', AGM_grid.discretization)


# ### Define the droplet pair as the initial condition

# In[21]:


N_DROPLETS = 2


# ### Separation distance between the droplet pair

# In[23]:


initial_separation = int(np.load('initial_separation.npy'))

# print('Initial_separation is', initial_separation)
#
# print()


# In[24]:


position_list = [[0, initial_separation/2, 0], [0, -initial_separation/2, 0]]

initial_radius_list_AGM = np.load('initial_radius_list_AGM.npy')


# In[25]:


list_of_droplets = [droplets.SphericalDroplet(position = position_list[i], radius = initial_radius_list_AGM[i])

                    for i in range(N_DROPLETS)]


# In[26]:


# list_of_droplets


# ### Initialize background equation with the right Boundary Conditions

# In[27]:


bc_x = [{'type': 'derivative', 'value': 0}, {'type': 'derivative', 'value': 0}]

bc_y = [{'type': 'derivative', 'value': 0}, {'type': 'derivative', 'value': 0}]

bc_z = [{'type': 'derivative', 'value': 0}, {'type': 'derivative', 'value': 0}]


# In[28]:


background = agent_based.DiffusionBackground({'diffusivity': DIFFUSION,
                                             'boundary_conditions': [bc_x, bc_y, bc_z]})
#
# background.show_parameters()
#
# print()


# ### Define the background scalarfield - uniform background

# In[29]:


initial_background = pde.ScalarField(AGM_grid, C_INF)


# In[30]:


# initial_background.plot(colorbar = True)

# plt.title('AGM-Initial Background Field')

# plt.close()


# ### Initialize agents

# ### Shell Thickness

# In[31]:


default_timestep_based_on_background = background.estimate_dt(initial_background)

# print('default_timestep_based_on_background is', default_timestep_based_on_background)
#
# print()


# In[32]:


# ### Shell Thickness based on Diffusivity

# shell_thickness_factor = int(CHANGE_shell_thickness_factor)

# shell_thickness = round((shell_thickness_factor * (np.sqrt(DIFFUSION * default_timestep_based_on_background))), 2)

######################################################################

# ### Shell Thickness based on Grid discretization

# shell_thickness = AGM_grid.discretization[0]

######################################################################

# ### Shell Thickness arbitrarily chosen

shell_thickness = round(CHANGE_shell_thickness, 1)

# shell_thickness = 1


# In[33]:

#
# print('FINAL shell_thickness is', shell_thickness)
#
# print()
#
# np.save('shell_thickness.npy', shell_thickness)


# ### Shell sector size

# In[34]:

shell_sector_size = round(CHANGE_shell_sector_size, 1)


# shell_sector_size = AGM_grid.discretization[0]

# shell_sector_size = int(1)


# In[35]:

#
# print('FINAL shell_sector_size is', shell_sector_size)
#
# print()
#
# np.save('shell_sector_size.npy', shell_sector_size)


# ######################################################################################################################

# In[36]:


agents = agent_based.SphericalDropletAgents({'equilibrium_concentration': str(2*GAMMA) + str('/radius'),
                                             'shell_thickness': shell_thickness,
                                             'shell_sector_size': shell_sector_size,
                                            'diffusivity': DIFFUSION,
                                            'drift_enabled': True,
                                           'reaction_outside': 0,
                                           'reaction_inside': 0})
#
# agents.show_parameters()
#
# print()


# ######################################################################################################################

# ### Setup simulation

# In[37]:


simulation = agent_based.AgentSimulation(background, agents)


# In[38]:


# simulation.info


# ### Make the final background_plus_agents

# In[39]:


background_plus_agents = simulation.get_state(background = initial_background, agents = list_of_droplets)


# In[40]:


# background_plus_agents.plot(filename = 'background_plus_agents.png', action = 'close')


# ### Make a MEGA file containing INFO

# In[41]:


# text_file = open("INFO-AGM.txt", "w")
#
# ######################################################################################################################
#
# text_file.write("\nN_DROPLETS_INITIAL is %d \n" %N_DROPLETS)
#
# ######################################################################################################################
#
# text_file.write("\n########################################")
#
# text_file.write("\n\nKAPPA is %f \n" %KAPPA)
#
# text_file.write("\nPREFACTOR_FREE_ENERGY_DENSITY is %f \n" %PREFACTOR_FREE_ENERGY_DENSITY)
#
# text_file.write("\nMOBILITY is %f \n" %MOBILITY)
#
# ######################################################################################################################
#
# text_file.write("\n########################################")
#
# text_file.write("\n\nGAMMA is %f \n" %GAMMA)
#
# text_file.write("\nInterface Width is %f \n" %W)
#
# if (C_INF > 0):
#
#     text_file.write("\nR_critical_theory is %f \n" %R_critical_theory)
#
#     text_file.write("\n########################################")
#
# ######################################################################################################################
#
# text_file.write("\n\nSYSTEM_SIZE for Agent Based Simulations is %f \n" %AGM_box_length)
#
# text_file.write("\n########################################")
#
# text_file.write("\n\ngrid discretization is %s \n" %str(AGM_grid.discretization))
#
# text_file.write("\nShell_thickness is %f \n" %float(shell_thickness))
#
# text_file.write("\nShell_sector_size is %f \n" %float(shell_sector_size))
#
# text_file.write("\n########################################\n")
#
# text_file.write("\nSample sim info is \n%s \n" %str(simulation.info))
#
# text_file.write("\n########################################")
#
# ######################################################################################################################
#
# text_file.close()


# ## Run the actual simulation

# #### Define TMAX

# In[42]:


# AGM_t_max = int(np.load('T_max.npy'))

AGM_t_max = int(6e3)

# print('AGM_t_max is', AGM_t_max)
#
# print()


# ### Define the trackers

# #### Tracking interval

# In[43]:


# desired_timepoints = int(1e2)
#
# tracking_interval =  int(AGM_t_max/desired_timepoints) # Frequency for writing out the data

tracking_interval =  1
# Frequency for writing out the data

# ### Droplet Tracker

# In[44]:


droplet_tracker = agent_based.DropletAgentTracker(interval = tracking_interval,
                                                  store_droplet_tracks = True,
                                                  store_emulsions = True)


# ### Background Tracker

# In[45]:


# storage = pde.FileStorage('AGM_background.hdf5')


# In[46]:


# background_storage_tracker = agent_based.BackgroundTracker(storage.tracker(tracking_interval))


# ### Run the simulation

# In[47]:


default_timestep = simulation.estimate_dt(background_plus_agents)

# print('default_timestep is', default_timestep)
#
# print()

aggressive_default_timestep = 0.05*simulation.estimate_dt(background_plus_agents)

# print('aggressive_default_timestep is', aggressive_default_timestep)
#
# print()


# In[48]:


# result = simulation.run(background_plus_agents, t_range = AGM_t_max,
#                         tracker = ['progress',
#                                    droplet_tracker,
#                                    background_storage_tracker])

########################################################################

result = simulation.run(background_plus_agents, t_range = AGM_t_max, dt = aggressive_default_timestep,
                        tracker = ['progress', droplet_tracker])

print()


# ### Plot the final background field

# In[49]:


# background_plus_agents.plot(title = 'Initial State', filename = 'Initial State.png', action = 'close')

# result.agents.plot(title = 'Final State', filename = 'Final State.png', action = 'close')


# ### Make a movie

# In[50]:


# visualization.movie_scalar(background_storage, 'AGM_2D_PassiveEmulsion.mov')


# ### Final droplet stats

# In[51]:


# result.agents.droplets.get_size_statistics()


# In[52]:


# if (result.agents.droplets.get_size_statistics()['radius_mean'] == 0):

#     raise ValueError('All droplets have disappeared')


# ### Save the droplet tracks and emulsions to a file

# In[53]:



AGM_droplet_tracks_filename = str('AGM_droplet_tracks_') + str('X=') + str(round(AGM_grid.discretization[0], 1)) + ',' + str('S=') + str(round(shell_sector_size, 1)) + ',' + str('L=') + str(round(shell_thickness, 1)) + '.hdf5'

droplet_tracker.droplet_tracks.to_file(AGM_droplet_tracks_filename)


# In[59]:


AGM_emulsions_filename = str('AGM_emulsions_') + str('X=') + str(round(AGM_grid.discretization[0], 1)) + ',' + str('S=') + str(round(shell_sector_size, 1)) + ',' + str('L=') + str(round(shell_thickness, 1)) + '.hdf5'

droplet_tracker.droplet_tracks.to_file(AGM_emulsions_filename)


# ######################################################################################################################

# ## Extract the radii and postitions of the droplets

# ### Identify global droplet recording timestamps

# In[55]:


global_droplet_timestamps = np.asarray(droplet_tracker.emulsions.times)


# ####################################################################################################################

# ### Isolate position and radius of the droplet

# In[56]:


single_droplet_timestamps = []

single_droplet_x_location = []

single_droplet_y_location = []

single_droplet_z_location = []

single_droplet_radius = []

for droplet in range(N_DROPLETS):

    single_droplet_track = droplet_tracker.droplet_tracks[droplet]

    single_droplet_track_info = single_droplet_track.data

    ########################################################################

    single_droplet_timestamps.append(single_droplet_track_info['time'])

    single_droplet_x_location.append(single_droplet_track_info['position'][:, 0])

    single_droplet_y_location.append(single_droplet_track_info['position'][:, 1])

    single_droplet_z_location.append(single_droplet_track_info['position'][:, 2])

    single_droplet_radius.append(single_droplet_track_info['radius'])

    ########################################################################


# In[57]:


single_droplet_timestamps = np.asarray(single_droplet_timestamps)

single_droplet_x_location = np.asarray(single_droplet_x_location)

single_droplet_y_location = np.asarray(single_droplet_y_location)

single_droplet_z_location = np.asarray(single_droplet_z_location)

single_droplet_radius = np.asarray(single_droplet_radius)

##### Pad appropriate zeros after dissolution events ######

single_droplet_radius_with_zeros = np.ones((N_DROPLETS, len(global_droplet_timestamps)))

for droplet in range(N_DROPLETS):

    number_of_zeros_needed = len(global_droplet_timestamps) - len(single_droplet_radius[droplet])

    zeros_array_to_be_padded = np.zeros((1, number_of_zeros_needed))

    single_droplet_radius_with_zeros[droplet, :] = np.append(single_droplet_radius[droplet], zeros_array_to_be_padded)

###################################################

# ### Plot the final droplet_tracks

# In[58]:


# for droplet in range(N_DROPLETS):

#     plt.plot(global_droplet_timestamps, single_droplet_radius[droplet], '--k', linewidth = 3)

#
# image_name = str('Tracks_AGM_') + str('X=') + str(round(AGM_grid.discretization[0], 2)) +             ',' + str('S=') + str(round(shell_sector_size, 2)) +             ',' + str('L=') + str(round(shell_thickness, 2)) + '.png'
#
#
# droplet_tracker.droplet_tracks.plot(linestyle = '-', color = 'k', linewidth = 2, alpha = 0.5,
#                                     title = str(3) + 'D Passive Droplet Pair, ' +
#                                     r'$L=$' + str(AGM_box_length) + ', ' +
#                                     r'$\Delta x=$' + str(round(AGM_grid.discretization[0], 2)) + ', ' +
#                                     r'$\Delta s=$' + str(round(shell_sector_size, 2)) + ', ' +
#                                     r'$\ell$=' + str(round(shell_thickness, 2)), fig_style = {'dpi': 200},
#                                     filename = image_name, action = 'close',
#                                     ax_style = {'xlim': (0, AGM_t_max),
#                                     'xlabel': r'$t$', 'ylabel': r'$\frac{R}{w}$'})


# ### Save all arrays

# In[60]:


# np.save('AGM_single_droplet_times.npy', single_droplet_timestamps)

# np.save('AGM_droplet_times.npy', global_droplet_timestamps)

####################################################################

# np.save('AGM_global_droplet_radius_original.npy', single_droplet_radius)

# np.save('AGM_droplet_radius.npy', single_droplet_radius)

# np.save('AGM_droplet_radius_with_zeros.npy', single_droplet_radius_with_zeros)
#
# ####################################################################
#
# np.save('AGM_droplet_x_location.npy', single_droplet_x_location)
#
# np.save('AGM_droplet_y_location.npy', single_droplet_y_location)
#
# np.save('AGM_droplet_z_location.npy', single_droplet_z_location)


# In[61]:

mean_radius = (single_droplet_radius_with_zeros[0] + single_droplet_radius_with_zeros[1])/2


# In[62]:


separation_distance = np.sqrt((single_droplet_x_location[0] - single_droplet_x_location[1])**2 +
                              (single_droplet_y_location[0] - single_droplet_y_location[1])**2 +
                              (single_droplet_z_location[0] - single_droplet_z_location[1])**2)


# In[63]:


filename_T = str('Time_') + str('X=') + str(round(AGM_grid.discretization[0], 1)) + ',' + str('S=') + str(round(shell_sector_size, 1)) + ',' + str('L=') + str(round(shell_thickness, 1)) + '.npy'

filename_mean_R = str('MeanRadius_') + str('X=') + str(round(AGM_grid.discretization[0], 1)) + ',' + str('S=') + str(round(shell_sector_size, 1)) + ',' + str('L=') + str(round(shell_thickness, 1)) + '.npy'

# filename_SD = str('SD_') + str('X=') + str(round(AGM_grid.discretization[0], 2)) + ',' + str('S=') + str(round(shell_sector_size, 2)) + ',' + str('L=') + str(round(shell_thickness, 2)) + '.npy'


# In[64]:


np.save(filename_T, global_droplet_timestamps)

np.save(filename_mean_R, mean_radius)

# np.save(filename_SD, separation_distance)


# ### Number of drops with time

# In[65]:

#
# global_droplet_number = []
#
# for snapshot in range(len(global_droplet_timestamps)):
#
#     global_droplet_number.append(len(droplet_tracker.emulsions.emulsions[snapshot]))
#
#
# # In[66]:
#
#
# np.save('AGM_droplet_number.npy', global_droplet_number)


# ### Plot Droplets vs Time

# In[67]:


# plt.plot(global_droplet_timestamps, global_droplet_number, linewidth = 3)
#
# plt.xlabel(r'$t$')
#
# plt.ylabel(r'$N_{drops}$')
#
# plt.minorticks_on()
# plt.grid(b = True, which='both', linewidth=0.05)
#
# plt.title(str(3) + 'D AGM Passive Droplet, ' + r'$L=$' + str(AGM_box_length) + ', ' +
#           str(N_DROPLETS) + ' Droplet, ' +
#           r'$\Delta x=$' + str(round(AGM_grid.discretization[0], 2)) + ', ' +
#           r'$\Delta s=$' + str(round(shell_sector_size, 2)) + ', ' +
#           r'$\ell$=' + str(round(shell_thickness, 2)))
#
# plt.xlim(0, AGM_t_max)
#
# plt.ylim(0, N_DROPLETS + 1)
#
# # plt.savefig('AGM-Droplets vs Time', dpi = 400, bbox_inches = 'tight')
#
# plt.close()


# In[68]:

#
# print('N_Drops initial is', global_droplet_number[0], ', N_Drops final is', global_droplet_number[-1])
#
# print()


# ### Plot mean radius and separation distance

# In[72]:


# CH_time = np.load('CH_droplet_times.npy')
#
# CH_MeanRadius = np.load('CH_MeanRadius.npy')
#
# plt.plot(CH_time, CH_MeanRadius, linestyle = '--', c = 'k', linewidth = 2, label = 'CH')
#
# plt.plot(global_droplet_timestamps, mean_radius, linewidth = 2, label = 'AGM')
#
# plt.xlabel(r'$t$')
# plt.ylabel(r'$\left \langle R \right \rangle$')
#
# # plt.minorticks_on()
# # plt.grid(b = True, which='both', linewidth=0.05)
#
# plt.title(str(3) + 'D Passive Droplet Pair, ' + r'$L=$' + str(AGM_box_length) + ', ' +
#           str(N_DROPLETS) + ' Droplet, ' +
#           r'$\Delta x=$' + str(round(AGM_grid.discretization[0], 2)) + ', ' +
#           r'$\Delta s=$' + str(round(shell_sector_size, 2)) + ', ' +
#           r'$\ell$=' + str(round(shell_thickness, 2)))
#
# plt.xlim(0, AGM_t_max)
#
# plt.legend(loc = 'upper right')
# plt.legend(frameon = False)
#
# image_name = str('MeanRadius_AGM_') + str('X=') + str(round(AGM_grid.discretization[0], 2)) +             ',' + str('S=') + str(round(shell_sector_size, 2)) +             ',' + str('L=') + str(round(shell_thickness, 2)) + '.png'
#
# plt.savefig(image_name, dpi = 200, bbox_inches = 'tight')
#
# plt.close()


# In[73]:

# CH_SeparationDistance = np.load('CH_SeparationDistance.npy')

# plt.plot(CH_time, CH_SeparationDistance, linestyle = '--', c = 'k', linewidth = 2)

# plt.plot(global_droplet_timestamps, separation_distance, linewidth = 2)

# plt.xlabel(r'$t$')
# plt.ylabel('Droplet separation')

# plt.minorticks_on()
# plt.grid(b = True, which='both', linewidth=0.05)

# plt.title(str(3) + 'D Passive Droplet Pair, ' + r'$L=$' + str(AGM_box_length) + ', ' +
#           str(N_DROPLETS) + ' Droplet, ' +
#           r'$\Delta x=$' + str(round(AGM_grid.discretization[0], 2)) + ', ' +
#           r'$\Delta s=$' + str(round(shell_sector_size, 2)) + ', ' +
#           r'$\ell$=' + str(round(shell_thickness, 2)))

# plt.xlim(0, AGM_t_max)

# image_name = str('SD_AGM_') + str('X=') + str(round(AGM_grid.discretization[0], 2)) +             ',' + str('S=') + str(round(shell_sector_size, 2)) +             ',' + str('L=') + str(round(shell_thickness, 2)) + '.png'

# plt.savefig(image_name, dpi = 400, bbox_inches = 'tight')

# plt.close()


# ####################################################################################################################

# In[71]:


# final_string_1 = 'rm '+ AGM_droplet_tracks_filename
#
# os.system(final_string_1)
#
# final_string_2 = 'rm '+ AGM_emulsions_filename
#
# os.system(final_string_2)
