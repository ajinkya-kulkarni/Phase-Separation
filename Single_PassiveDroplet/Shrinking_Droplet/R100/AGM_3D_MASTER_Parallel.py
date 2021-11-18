#!/usr/bin/env python
# coding: utf-8

# # Single Passive droplet in a saturated environment. For studying the discretization effects

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math

import os

# ### Importing the code + PyPackage in the cluster

# In[2]:


############################## BEGIN CLUSTER STUFF ##############################

# Preserve environment variables
#$ -V

# Use python as shell
#$ -S /home/ajinkya/anaconda3/bin/python3

# Execute from current working directory
#$ -cwd

##############################

file = open("/home/ajinkya/PyPackage_from_Github/PyPackagePath.txt", "r") # Cluster version

############################## END CLUSTER STUFF ##############################


# ### Importing the code + PyPackage locally

# In[3]:


# file = open("/Users/ajinkyakulkarni/Desktop/Droplets-Package/PyPackagePath.txt", "r") # Local laptop version

##############################

path_location = str(file.read())

path_location = path_location.splitlines()

##############################

# Check if the paths are printed properly

# for paths in path_location:
#
#     print()
#     print(paths)
#
# print()


# In[4]:


import sys


# In[5]:


for paths in path_location:

    sys.path.append(paths)


# ### Import the packages from the PyPackage

# In[6]:


import pde

import droplets

import phasesep

import agent_based


# ######################################################################################################################

# In[7]:


# %config InlineBackend.figure_format ='retina'

# plt.rcParams.update({'font.size': 15})


# In[8]:


import h5py


# In[9]:


import traceback


# ## Define physical parameters for the code

# In[10]:


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


# In[11]:


DIFFUSION = PREFACTOR_FREE_ENERGY_DENSITY * MOBILITY

K0 = DIFFUSION/(W**2)


# In[12]:


file = open("C_INF.out", "r")

C_INF = float(file.read())

# print('C_INF is ', C_INF)


# In[13]:

if C_INF > 0:

    file = open("R_critical_theory.out", "r")
    R_critical_theory = float(file.read())

    # R_critical is almost equal to R_critical_PRE and R_critical_FINITE and R_critical_INFINITE

    print('Critical radius is ', R_critical_theory)

file = open("AGM_SYSTEM_SIZE.out", "r")

AGM_SYSTEM_SIZE = int(file.read())

# ##################################################################
# ### Define the 3D Cartesian grid - for Agent based Model simulations

# In[16]:

# number_support_points_GRID = int(AGM_SYSTEM_SIZE/CHANGE_grid_discretization)
#
# ##################################################################
#
# if ((number_support_points_GRID % 2) != 0):
#
#     number_support_points_GRID = number_support_points_GRID + 1


# In[17]:


# print('Support_points for AGM grid are', number_support_points_GRID)
# print()
# print('System size for AGM is', AGM_SYSTEM_SIZE)
# print()


# In[18]:


# np.save('number_support_points_GRID.npy', number_support_points_GRID)

# In[18]:

AGM_grid = pde.CartesianGrid([[-AGM_SYSTEM_SIZE/2, AGM_SYSTEM_SIZE/2],
                              [-AGM_SYSTEM_SIZE/2, AGM_SYSTEM_SIZE/2],
                              [-AGM_SYSTEM_SIZE/2, AGM_SYSTEM_SIZE/2]],
                             [int(CHANGE_support_points_GRID), int(CHANGE_support_points_GRID), int(CHANGE_support_points_GRID)],
                             periodic = [False, False, False])


# In[19]:

# print('AGM_grid_discretization is', AGM_grid.discretization)

# print()

# np.save('AGM_grid_discretization.npy', AGM_grid.discretization)


# ### Define the single droplet as the initial condition

# In[20]:


N_DROPLETS = 1


# In[21]:


initial_radius = float(np.load('initial_radius_AGM.npy'))

# print('initial_radius is', initial_radius)
# print()


# In[22]:


list_of_droplets = [droplets.SphericalDroplet(position = [0, 0, 0], radius = initial_radius)

                    for i in range(N_DROPLETS)]


# ### Initialize background equation with the right Boundary Conditions

# In[23]:


bc_x = [{'value': C_INF}, {'value': C_INF}]

bc_y = [{'value': C_INF}, {'value': C_INF}]

bc_z = [{'value': C_INF}, {'value': C_INF}]


# In[24]:


background = agent_based.DiffusionBackground({'diffusivity': DIFFUSION,
                                             'boundary_conditions': [bc_x, bc_y, bc_z]})

# background.show_parameters()

# print()


# ### Define the background scalarfield - uniform background

# In[25]:


# initial_background = pde.ScalarField(AGM_grid, C_INF)


# In[26]:


# initial_background.plot(colorbar = True)

# plt.title('AGM-Initial Background Field')

# # plt.close()


# ### Define the background scalarfield - self defined

# In[27]:


# Outside concentration profile according to IN-FINITE box analytics

x_axis = np.linspace(-AGM_SYSTEM_SIZE/2, AGM_SYSTEM_SIZE/2, int(CHANGE_support_points_GRID))
y_axis = np.linspace(-AGM_SYSTEM_SIZE/2, AGM_SYSTEM_SIZE/2, int(CHANGE_support_points_GRID))
z_axis = np.linspace(-AGM_SYSTEM_SIZE/2, AGM_SYSTEM_SIZE/2, int(CHANGE_support_points_GRID))

xx, yy, zz = np.meshgrid(x_axis, y_axis, z_axis)

############################################################

radial_axis = np.sqrt(xx*xx + yy*yy + zz*zz)


# In[28]:


C_out = -initial_radius*C_INF/radial_axis + 2*GAMMA/radial_axis + C_INF

C_in_AND_C_out = np.where(radial_axis < initial_radius, 2*GAMMA/initial_radius, C_out)


# #### Convert into a ScalarField

# In[29]:


initial_background = pde.ScalarField(AGM_grid, C_in_AND_C_out)


# ### Smooth the background field

# In[30]:


initial_background = initial_background.smooth()


# In[31]:


# plt.plot(x_axis, initial_background.get_line_data(extract = 'cut_x')['data_y'], label=r'$x-axis$')
#
# plt.plot(y_axis, initial_background.get_line_data(extract = 'cut_y')['data_y'], label=r'$y-axis$')
#
# plt.plot(z_axis, initial_background.get_line_data(extract = 'cut_z')['data_y'], label=r'$z-axis$')
#
# # #########################################################
#
# plt.axhline(y = C_INF, linestyle = '-.', c = 'k', linewidth = 1, label=r'$C_{\infty}$')
#
# plt.axhline(y = 2*GAMMA/initial_radius, linestyle = '--', c = 'k', linewidth = 1, label=r'$C_{eq}^{out}$')
#
# # #########################################################
#
# plt.minorticks_on()
# plt.grid(b=True, which='both', linewidth=0.05)
#
# plt.legend(loc = 'best')
#
# plt.ylabel('$C_{out}$')
#
# plt.xlabel('Co-ordinate')
#
# plt.xlim(-AGM_SYSTEM_SIZE/2, AGM_SYSTEM_SIZE/2)
#
# plt.title('AGM-Initial Background Field')
#
# plt.savefig('AGM-Initial Background Field', dpi = 400, bbox_inches = 'tight')
#
# #########################################################
#
# plt.close()


# ### Initialize agents

# ### Shell Thickness based on diffusivity

# In[32]:


default_timestep_based_on_background = background.estimate_dt(initial_background)

# print('default_timestep_based_on_background is', default_timestep_based_on_background)
#
# print()


# In[33]:

# ### Shell Thickness based on Diffusivity

# shell_thickness = float(CHANGE_shell_thickness_factor) * (np.sqrt(DIFFUSION * default_timestep_based_on_background))

######################################################################

# ### Shell Thickness based on Grid discretization

# shell_thickness = AGM_grid.discretization[0]

######################################################################

# ### Shell Thickness based on droplet spacing

# shell_thickness = 10*float(np.load('mean_droplet_surface_separation.npy'))

######################################################################

# ### Shell Thickness arbitrarily chosen

shell_thickness = round(CHANGE_shell_thickness, 1)

# shell_thickness = 1

# ######################################################################################################################
#
# print('FINAL shell_thickness is', shell_thickness)
#
# print()

# np.save('shell_thickness.npy', shell_thickness)


# ##### Change shell sector size

shell_sector_size = round(CHANGE_shell_sector_size, 1)

########################################################################################

# shell_sector_size = AGM_grid.discretization[0]

########################################################################################

# shell_sector_size = int(1)
#
# print('FINAL shell_sector_size is', shell_sector_size)
#
# print()

# np.save('shell_sector_size.npy', shell_sector_size)

# ######################################################################################################################

# In[42]:


agents = agent_based.SphericalDropletAgents({'equilibrium_concentration': str(2*GAMMA) + str('/radius'),
                                             'shell_thickness': shell_thickness,
                                             'shell_sector_size': shell_sector_size,
                                            'diffusivity': DIFFUSION,
                                            'drift_enabled': True,
                                           'reaction_outside': 0,
                                           'reaction_inside': 0})

# agents.show_parameters()

# print()


# ######################################################################################################################

# ### Setup simulation

# In[43]:


simulation = agent_based.AgentSimulation(background, agents)


# In[44]:


# simulation.info


# ### Make the final background_plus_agents

# In[45]:


background_plus_agents = simulation.get_state(background = initial_background, agents = list_of_droplets)


# In[46]:


# background_plus_agents.plot()

# plt.title('AGM-Initial Background Field + Agents')

# # plt.savefig('AGM-Initial state', dpi = 400, bbox_inches = 'tight')

# # plt.close()


# ### Make a MEGA file containing INFO

# In[47]:

#
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
# text_file.write("\nEquilibrium concentration is %s \n" %(str(2*GAMMA) + str('/radius')))
#
# text_file.write("\nC_INF is %f \n" %C_INF)
#
# if C_INF > 0:
#
#     Critical_radius = 2*GAMMA/C_INF
#
#     text_file.write("\nR_critical_theory is %f \n" %R_critical_theory)
#
# text_file.write("\n########################################")
#
# ######################################################################################################################
#
# text_file.write("\n\nSYSTEM_SIZE for Agent Based Simulations is %f \n" %AGM_SYSTEM_SIZE)
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
#
#
# #
# ## Run the actual simulation

# #### Define TMAX

# In[48]:


AGM_t_max = int(np.load('T_max.npy'))


# In[34]:


# print('AGM_t_max is', AGM_t_max)
#
# print()


# ### Define the trackers

# #### Tracking interval

# In[49]:


desired_timepoints = int(1e2)

tracking_interval =  int(AGM_t_max/desired_timepoints) # Frequency for writing out the data

# print('AGM_t_max is', AGM_t_max, ', Desired timepoints are', desired_timepoints,
#       'and tracking interval is', tracking_interval)

# ### Droplet Tracker

# In[50]:


droplet_tracker = agent_based.DropletAgentTracker(interval = tracking_interval,
                                                  store_droplet_tracks = True,
                                                  store_emulsions = True)


# ### Background Tracker

# In[51]:


# storage = pde.FileStorage('AGM_background.hdf5')


# In[52]:


# background_storage_tracker = agent_based.BackgroundTracker(storage.tracker(tracking_interval))


# ### Run the simulation

# ##### Define timestep

# In[53]:


default_timestep = simulation.estimate_dt(background_plus_agents)
#
# print('default_timestep is', default_timestep)
#
# print()

aggressive_default_timestep = 0.1*simulation.estimate_dt(background_plus_agents)

# print('aggressive_default_timestep is', aggressive_default_timestep)
#
# print()


# In[54]:


# result = simulation.run(background_plus_agents, t_range = AGM_t_max,
#                         tracker = ['progress',
#                                    droplet_tracker,
#                                    background_storage_tracker])

########################################################################

result = simulation.run(background_plus_agents, t_range = AGM_t_max, dt = aggressive_default_timestep,
                        tracker = ['progress', droplet_tracker])

print()


# final_background_structure = result.background.data
#
# filename_concentration_profile = str('Concentration_profile_background_') + str('X=') + str(round(AGM_grid.discretization[0], 1)) + ',' + str('S=') + str(round(shell_sector_size, 1)) + ',' + str('L=') + str(round(shell_thickness, 1)) + '.npy'

#np.save(filename_concentration_profile, final_background_structure)

# ### Make a movie

# In[55]:


# visualization.movie_scalar(background_storage, 'AGM_2D_PassiveEmulsion.mov')


# ### Final droplet stats

# In[56]:


# result.agents.droplets.get_size_statistics()


# In[57]:


#if (result.agents.droplets.get_size_statistics()['radius_mean'] == 0):
#
#    raise ValueError('All droplets have disappeared')


# ### Save the droplet tracks and emulsions to a file

# In[58]:


AGM_droplet_tracks_filename = str('AGM_droplet_tracks_') + str('X=') + str(round(AGM_grid.discretization[0], 1)) + ',' + str('S=') + str(round(shell_sector_size, 1)) + ',' + str('L=') + str(round(shell_thickness, 1)) + '.hdf5'

droplet_tracker.droplet_tracks.to_file(AGM_droplet_tracks_filename)


# In[59]:


AGM_emulsions_filename = str('AGM_emulsions_') + str('X=') + str(round(AGM_grid.discretization[0], 1)) + ',' + str('S=') + str(round(shell_sector_size, 1)) + ',' + str('L=') + str(round(shell_thickness, 1)) + '.hdf5'

droplet_tracker.droplet_tracks.to_file(AGM_emulsions_filename)


# ### Plot orignal droplet tracks

# In[60]:


# droplet_tracker.droplet_tracks.plot(linewidth = 2,
#                                     title = str(3) + 'D Passive Droplet, ' + r'$L=$' + str(AGM_SYSTEM_SIZE) + ', ' +
#                                     str(N_DROPLETS) + ' Droplet, ' +
#                                     r'$\Delta x=$' + str(round(AGM_grid.discretization[0], 2)) + ', ' +
#                                     r'$\Delta s=$' + str(round(shell_sector_size, 2)) + ', ' +
#                                     r'$\ell$=' + str(round(shell_thickness, 2)), fig_style = {'dpi': 200})

# # plt.minorticks_on()
# # plt.grid(b = True, which='both',linewidth=0.05)


# ######################################################################################################################

# ## Extract the radii and postitions of the droplets

# ### Identify global droplet recording timestamps

# In[61]:


global_droplet_timestamps = np.asarray(droplet_tracker.emulsions.times)

# ####################################################################################################################

# ### Isolate position and radius of the droplet

# In[62]:


single_droplet_timestamps = []

#single_droplet_x_location = []
#
#single_droplet_y_location = []
#
#single_droplet_z_location = []

single_droplet_radius = []

for droplet in range(N_DROPLETS):

    single_droplet_track = droplet_tracker.droplet_tracks[droplet]

    single_droplet_track_info = single_droplet_track.data

    ########################################################################

    single_droplet_timestamps.append(single_droplet_track_info['time'])

#    single_droplet_x_location.append(single_droplet_track_info['position'][:, 0])
#
#    single_droplet_y_location.append(single_droplet_track_info['position'][:, 1])
#
#    single_droplet_z_location.append(single_droplet_track_info['position'][:, 2])

    single_droplet_radius.append(single_droplet_track_info['radius'])

    ########################################################################


# In[63]:


single_droplet_timestamps = np.asarray(single_droplet_timestamps)

#single_droplet_x_location = np.asarray(single_droplet_x_location)
#
#single_droplet_y_location = np.asarray(single_droplet_y_location)
#
#single_droplet_z_location = np.asarray(single_droplet_z_location)

single_droplet_radius = np.asarray(single_droplet_radius)

##### Pad appropriate zeros after dissolution events ######

single_droplet_radius_with_zeros = np.ones((N_DROPLETS, len(global_droplet_timestamps)))

for droplet in range(N_DROPLETS):

    number_of_zeros_needed = len(global_droplet_timestamps) - len(single_droplet_radius[droplet])

    zeros_array_to_be_padded = np.zeros((1, number_of_zeros_needed))

    single_droplet_radius_with_zeros[droplet, :] = np.append(single_droplet_radius[droplet], zeros_array_to_be_padded)

###################################################

# In[64]:


# np.save('R_simulations_AGM.npy', single_droplet_radius_with_zeros[0][-1])

# print('Initial droplet radius was:', single_droplet_radius_with_zeros[0][0],
#       'and final droplet radius is:', round(single_droplet_radius_with_zeros[0][-1], 2))
#
# print()


# ### Plot the final droplet_tracks

# In[65]:


# for droplet in range(N_DROPLETS):
#
#     plt.plot(global_droplet_timestamps, single_droplet_radius_with_zeros[droplet], '--k', linewidth = 3)
#
# CH_radius = np.load('CH_droplet_radius.npy')[0]
#
# CH_time = np.load('CH_droplet_times.npy')
#
# plt.plot(CH_time, CH_radius, 'steelblue', linewidth = 3, label = 'CH')
#
# plt.ylabel('$\frac{R}{w}$')
#
# plt.ylabel('$t$')
#
# plt.legend()
#
# image_name = str('AGM_') + str('X=') + str(round(AGM_grid.discretization[0], 2)) +             ',' + str('S=') + str(round(shell_sector_size, 2)) +             ',' + str('L=') + str(round(shell_thickness, 2)) + '.png'
#
# plt.title(r'$\Delta x=$' + str(round(AGM_grid.discretization[0], 1)) + 'w, ' +
#                                     r'$\Delta s=$' + str(round(shell_sector_size, 1)) + 'w, ' +
#                                     r'$\ell$=' + str(round(shell_thickness, 1)) + 'w')
#
# plt.savefig(image_name, dpi = 400, bbox_inches = 'tight')
#
# plt.close()

# image_name = str('AGM_') + str('X=') + str(round(AGM_grid.discretization[0], 2)) +             ',' + str('S=') + str(round(shell_sector_size, 2)) +             ',' + str('L=') + str(round(shell_thickness, 2)) + '.png'


# droplet_tracker.droplet_tracks.plot(linestyle = '-', color = 'k', linewidth = 2, alpha = 0.5,
#                                     title = str(3) + 'D Passive Droplet, ' + r'$L=$' + str(AGM_SYSTEM_SIZE) + ', ' +
#                                     str(N_DROPLETS) + ' Droplet, ' +
#                                     r'$\Delta x=$' + str(round(AGM_grid.discretization[0], 2)) + ', ' +
#                                     r'$\Delta s=$' + str(round(shell_sector_size, 2)) + ', ' +
#                                     r'$\ell$=' + str(round(shell_thickness, 2)), fig_style = {'dpi': 200},
#                                     filename = image_name, action = 'close',
#                                     ax_style = {'xlim': (0, AGM_t_max),
#                                               'ylim': (-0.5, 1.1 * single_droplet_radius[0][-1]),
#                                     'xlabel': r'$t$', 'ylabel': r'$\frac{R}{w}$'})


# ### Plot location

# In[66]:


# ### Save all arrays

# In[67]:


# np.save('AGM_single_droplet_times.npy', single_droplet_timestamps)

# np.save('AGM_droplet_times.npy', global_droplet_timestamps)

####################################################################

# np.save('AGM_global_droplet_radius_original.npy', single_droplet_radius)

#np.save('AGM_droplet_radius.npy', single_droplet_radius)

# np.save('AGM_droplet_radius_with_zeros.npy', single_droplet_radius_with_zeros)

####################################################################

#np.save('AGM_droplet_x_location.npy', single_droplet_x_location)
#
#np.save('AGM_droplet_y_location.npy', single_droplet_y_location)
#
#np.save('AGM_droplet_z_location.npy', single_droplet_z_location)


# In[68]:


filename_T = str('Time_') + str('X=') + str(round(AGM_grid.discretization[0], 1)) + ',' + str('S=') + str(round(shell_sector_size, 1)) + ',' + str('L=') + str(round(shell_thickness, 1)) + '.npy'

filename_R = str('Radius_') + str('X=') + str(round(AGM_grid.discretization[0], 1)) + ',' + str('S=') + str(round(shell_sector_size, 1)) + ',' + str('L=') + str(round(shell_thickness, 1)) + '.npy'

# In[69]:


np.save(filename_T, global_droplet_timestamps)

np.save(filename_R, single_droplet_radius_with_zeros)


# ### Number of drops with time

# In[70]:


# global_droplet_number = []
#
# for snapshot in range(len(global_droplet_timestamps)):
#
#     global_droplet_number.append(len(droplet_tracker.emulsions.emulsions[snapshot]))


# In[71]:


# np.save('AGM_droplet_number.npy', global_droplet_number)


# ### Plot Droplets vs Time

# In[72]:

#
# plt.plot(global_droplet_timestamps, global_droplet_number, linewidth = 3)
#
# plt.xlabel(r'$t$')
#
# plt.ylabel(r'$N_{drops}$')
#
# plt.minorticks_on()
# plt.grid(b = True, which='both', linewidth=0.05)
#
# plt.title(str(3) + 'D AGM Passive Droplet, ' + r'$L=$' + str(AGM_SYSTEM_SIZE) + ', ' +
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


# In[73]:


# print('N_Drops initial is', global_droplet_number[0], ', N_Drops final is', global_droplet_number[-1])
#
# print()


# ### Save the concentration profile of the background of the final state

# In[74]:


# plt.plot(x_axis, result.background.field.get_line_data(extract = 'cut_x')['data_y'],
#          color = 'darkorange', label=r'$t=T_{max}$')
#
# ####################################################################
#
# plt.axhline(y = C_INF, linestyle = '--', c = 'g', linewidth = 1, label=r'$C_{\infty}$')
#
# ####################################################################
#
# #plt.axhline(y = 2*GAMMA/single_droplet_radius[0][-1], c = 'darkorange', linestyle = '-.', linewidth = 1,
# #            label=r'$C_{eq}^{out}, t=T_{max}$')
#
# ####################################################################
#
# plt.plot(x_axis, initial_background.get_line_data(extract = 'cut_x')['data_y'],
#          color = 'steelblue', label=r'$t=0$')
#
# ####################################################################
#
# plt.axhline(y = 2*GAMMA/initial_radius, c = 'steelblue', linestyle = '-.', linewidth = 1,
#             label=r'$C_{eq}^{out}, t=0$')
#
# ####################################################################
#
# plt.minorticks_on()
# plt.grid(b=True, which='both', linewidth=0.05)
#
# plt.legend(loc = 'center right', bbox_to_anchor = [1.5, 0.5])
#
# plt.ylabel('$C_{out}$')
#
# plt.xlabel('Co-ordinate')
#
# plt.xlim(-AGM_SYSTEM_SIZE/2, AGM_SYSTEM_SIZE/2)
#
# plt.title('AGM-Background Field')
#
# plt.close()


# In[75]:


# AGM_background_concentration_support_points = result.background.field.get_line_data()['data_x']
#
# AGM_background_concentration_profile = result.background.field.get_line_data()['data_y']
#
# filename_support_points = str('support_points_') + str('X=') + str(round(AGM_grid.discretization[0], 1)) + ',' + str('S=') + str(round(shell_sector_size, 1)) + ',' + str('L=') + str(round(shell_thickness, 1)) + '.npy'
#
# filename_concentration_profile = str('concentration_profile_') + str('X=') + str(round(AGM_grid.discretization[0], 1)) + ',' + str('S=') + str(round(shell_sector_size, 1)) + ',' + str('L=') + str(round(shell_thickness, 1)) + '.npy'
#
# # In[69]:


#np.save(filename_support_points, AGM_background_concentration_support_points)
#
#np.save(filename_concentration_profile, AGM_background_concentration_profile)

# ####################################################################################################################

# In[77]:


# In[59]:

# final_string_1 = 'rm '+ AGM_droplet_tracks_filename
#
# os.system(final_string_1)
#
# final_string_2 = 'rm '+ AGM_emulsions_filename
#
# os.system(final_string_2)

# In[78]:


# results_array = [initial_radius, AGM_SYSTEM_SIZE, number_support_points_GRID,
#                  shell_thickness, shell_sector_size,
#                  single_droplet_radius[0][-1]]

# np.savetxt("RESULTS.txt", [results_array], delimiter='\t', fmt = "%0.30f")


# In[ ]:
