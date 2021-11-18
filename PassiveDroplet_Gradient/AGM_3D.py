# !/usr/bin/env python
# coding: utf-8

# # Passive Droplet in a chemical gradient in an $\infty ~3D$ domain using Agent based Modelling using a Cartesian grid

# ######################################################################################################################

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math


# In[2]:


import os


# In[3]:


# plt.rcParams.update({'font.size': 15})


# In[5]:


import h5py


import traceback


############################## BEGIN CLUSTER STUFF ##############################

# # Preserve environment variables
# #$ -V
#
# # Use python as shell
# #$ -S /home/ajinkya/anaconda3/bin/python3
#
# # Execute from current working directory
# #$ -cwd
#
# # #############################
# #
# file = open("/home/ajinkya/PyPackage_from_Github/PyPackagePath.txt", "r") # Cluster version
#
# path_location = str(file.read())
# path_location = path_location.splitlines()
#
# import sys
# for paths in path_location:
#     sys.path.append(paths)
#
# import pde, droplets, phasesep, agent_based

############################# END CLUSTER STUFF ##############################

import sys

sys.path.append('/Users/ajinkyakulkarni/Desktop/GitHub/py-pde')
sys.path.append('/Users/ajinkyakulkarni/Desktop/GitHub/py-droplets')
sys.path.append('/Users/ajinkyakulkarni/Desktop/GitHub/py-phasesep')
sys.path.append('/Users/ajinkyakulkarni/Desktop/GitHub/agent-based-emulsions')

import pde, droplets, phasesep, agent_based
# ######################################################################################################################

# #### Tracking interval

# In[12]:


# tracking_interval = int(np.load('tracking_interval.npy')) # Frequency for writing out the data

#print('tracking_interval is', tracking_interval)
#
#print()


# ### Read the parameters

# In[13]:


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


# In[14]:


file = open("alpha.out", "r")
ALPHA = float(file.read())

file = open("beta.out", "r")
BETA = float(file.read())


# In[15]:


file = open("limit_r.out", "r")
limit_r = int(file.read())

file = open("limit_z.out", "r")
limit_z = int(file.read())

file = open("AGM_SYSTEM_SIZE.out", "r")
AGM_SYSTEM_SIZE = int(file.read())

############################################################

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


# In[16]:


DIFFUSION = PREFACTOR_FREE_ENERGY_DENSITY * MOBILITY

K0 = DIFFUSION/(W**2)


# In[17]:


file = open("R_critical_theory.out", "r")

R_critical_theory = float(file.read())

#print('Critical radius is', R_critical_theory)
#print()


# ##################################################################
# ### Define the 3D Cartesian grid - for Agent based Model simulations

# In[16]:

initial_radius = int(np.load('initial_radius_AGM.npy'))

check_axis = np.linspace(-AGM_SYSTEM_SIZE/2, AGM_SYSTEM_SIZE/2, int(AGM_SYSTEM_SIZE/initial_radius))

if (np.any(check_axis == 0) == True):

    gridpoints_actual = len(check_axis) + 1

else:

    gridpoints_actual = len(check_axis)

AGM_grid = pde.CartesianGrid([[-AGM_SYSTEM_SIZE/2, AGM_SYSTEM_SIZE/2],
                              [-AGM_SYSTEM_SIZE/2, AGM_SYSTEM_SIZE/2],
                              [-AGM_SYSTEM_SIZE/2, AGM_SYSTEM_SIZE/2]],
                             [int(gridpoints_actual), int(gridpoints_actual), int(gridpoints_actual)],
                             periodic = [False, False, False])

# In[23]:


print('AGM_grid_discretization is', AGM_grid.discretization)
print()
#
#np.save('AGM_grid_discretization.npy', AGM_grid.discretization)


# ### Define the single droplet as the initial condition

# In[24]:


N_DROPLETS = 1

list_of_droplets = [droplets.SphericalDroplet(position = [0, 0, 0], radius = initial_radius)

                    for i in range(N_DROPLETS)]


# ### Initialize background equation with the right Boundary Conditions

# In[27]:


bc_x = [{'type': 'derivative', 'value': 0}, {'type': 'derivative', 'value': 0}]

bc_y = [{'type': 'value', 'value': C_bottom_y_AGM}, {'type': 'value', 'value': C_top_y_AGM}]

bc_z = [{'type': 'derivative', 'value': 0}, {'type': 'derivative', 'value': 0}]


# In[28]:


background = agent_based.DiffusionBackground({'diffusivity': DIFFUSION,
                                             'boundary_conditions': [bc_x, bc_y, bc_z]})
#
#background.show_parameters()
#
#print()


# ### Define the background scalarfield - uniform background

# In[29]:


# initial_background = pde.ScalarField(AGM_grid, C_INF)


# In[30]:


# initial_background.plot(colorbar = True)

# plt.title('AGM-Initial Background Field')

# # plt.close()


# ### Define the background scalarfield - self defined

# In[31]:


x_axis = np.linspace(-AGM_SYSTEM_SIZE/2, AGM_SYSTEM_SIZE/2, int(gridpoints_actual))

y_axis = x_axis

z_axis = x_axis

# In[32]:


C_in_AND_C_out = np.zeros((int(gridpoints_actual), int(gridpoints_actual), int(gridpoints_actual)))

for i in range(int(gridpoints_actual)):

    for j in range(int(gridpoints_actual)):

        for k in range(int(gridpoints_actual)):

            r = np.sqrt(x_axis[i]*x_axis[i] + y_axis[j]*y_axis[j] + z_axis[k]*z_axis[k])

            phi = math.acos(z_axis[k]/r) # Refer to Spherical to Cartesian co-ordinate transformation

            if (r > initial_radius):

                C_in_AND_C_out[i, k, j] = ALPHA*(-initial_radius/r + 1) + BETA*(-initial_radius**3/r**2 + r)*np.cos(phi) + 2*GAMMA/r

            else:

                C_in_AND_C_out[i, k, j] = 2*GAMMA/initial_radius


# #### Convert into a ScalarField

# In[33]:


initial_background = pde.ScalarField(AGM_grid, C_in_AND_C_out)


# ### Smooth the background field

# In[34]:


initial_background = initial_background.smooth()


# In[35]:


#plt.plot(x_axis, initial_background.get_line_data(extract = 'cut_x')['data_y'], label=r'$x-axis$')
#
#plt.plot(y_axis, initial_background.get_line_data(extract = 'cut_y')['data_y'], label=r'$y-axis$')
#
#plt.plot(z_axis, initial_background.get_line_data(extract = 'cut_z')['data_y'], label=r'$z-axis$')
#
## #########################################################
#
#plt.axhline(y = 2*GAMMA/initial_radius, linestyle = '--', c = 'k', linewidth = 1, label=r'$C_{eq}^{out}$')
#
#linear_gradient = np.linspace(C_bottom_y_AGM, C_top_y_AGM, number_support_points_GRID)
#
#plt.plot(y_axis, linear_gradient, '--', label = 'Linear gradient w.r.t local Droplet centre')
#
#plt.axvline(x = 0, c = 'k', linestyle = '-.', linewidth = 1, label = 'Droplet centre')
#
## #########################################################
#
#plt.minorticks_on()
#plt.grid(b=True, which='both', linewidth=0.05)
#
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#
#plt.ylabel('$C_{out}$')
#
#plt.xlabel('Co-ordinate')
#
#plt.xlim(-AGM_SYSTEM_SIZE/10, AGM_SYSTEM_SIZE/10)
#
#plt.title('AGM-Initial Background Field')
#
#plt.savefig('AGM-Initial Background Field', dpi = 400, bbox_in_es = 'tight')
#
#plt.close()


# In[36]:


#initial_background.plot(title = 'AGM-Initial Background Field',
#                        colorbar = 'True', vmin = C_bottom_y_AGM, vmax = C_top_y_AGM, action = 'close')
#
# plt.savefig('AGM-Background_Field-image', dpi = 400, bbox_inches = 'tight')

# plt.close()


# ### Initialize agents

# ### Shell Thickness based on diffusivity

# In[37]:


default_timestep_based_on_background = background.estimate_dt(initial_background)
#
#print('default_timestep_based_on_background is', default_timestep_based_on_background)
#
#print()


# In[38]:


# ### Shell Thickness based on Diffusivity

# shell_thickness = float(CHANGE_shell_thickness_factor) * (np.sqrt(DIFFUSION * default_timestep_based_on_background))

######################################################################

# ### Shell Thickness based on Grid discretization

shell_thickness = float(AGM_grid.discretization[0])

######################################################################

# ### Shell Thickness based on droplet spacing

# shell_thickness = 10*float(np.load('mean_droplet_surface_separation.npy'))

######################################################################

# ### Shell Thickness arbitrarily chosen

# shell_thickness = int(CHANGE_shell_thickness)

# shell_thickness = 1

# ######################################################################################################################

#print('FINAL shell_thickness is', shell_thickness)
#
#print()
#
#np.save('shell_thickness.npy', shell_thickness)


# ##### Change shell sector size

# shell_sector_size = int(CHANGE_shell_sector_size)

########################################################################################

shell_sector_size = float(AGM_grid.discretization[0])

########################################################################################

# shell_sector_size = int(1)

#print('FINAL shell_sector_size is', shell_sector_size)
#
#print()
#
#np.save('shell_sector_size.npy', shell_sector_size)


# ######################################################################################################################

# In[45]:


agents = agent_based.SphericalDropletAgents({'equilibrium_concentration': str(2*GAMMA) + str('/radius'),
                                             'shell_thickness': shell_thickness,
                                             'shell_sector_size': shell_sector_size,
                                            'diffusivity': DIFFUSION,
                                            'drift_enabled': True,
                                           'reaction_outside': 0,
                                           'reaction_inside': 0})
#
#agents.show_parameters()
#
#print()
#

# ######################################################################################################################

# ### Setup simulation

# In[46]:


simulation = agent_based.AgentSimulation(background, agents)


# In[47]:


#simulation.info


# ### Make the final background_plus_agents

# In[48]:


background_plus_agents = simulation.get_state(background = initial_background, agents = list_of_droplets)


# ### Make a MEGA file containing INFO

# In[49]:


text_file = open("INFO-AGM.txt", "w")

######################################################################################################################

text_file.write("\nN_DROPLETS_INITIAL is %d \n" %N_DROPLETS)

######################################################################################################################

text_file.write("\n########################################")

text_file.write("\n\nKAPPA is %f \n" %KAPPA)

text_file.write("\nPREFACTOR_FREE_ENERGY_DENSITY is %f \n" %PREFACTOR_FREE_ENERGY_DENSITY)

text_file.write("\nMOBILITY is %f \n" %MOBILITY)

######################################################################################################################

text_file.write("\n########################################")

text_file.write("\n\nGAMMA is %f \n" %GAMMA)

text_file.write("\nInterface Width is %f \n" %W)

text_file.write("\nR_critical_theory is %f \n" %R_critical_theory)

text_file.write("\n########################################")

######################################################################################################################

text_file.write("\n\nSYSTEM_SIZE for Agent Based Simulations is %f \n" %AGM_SYSTEM_SIZE)

text_file.write("\n########################################")

text_file.write("\n\ngrid discretization is %s \n" %str(AGM_grid.discretization))

text_file.write("\nShell_thickness is %f \n" %float(shell_thickness))

text_file.write("\nShell_sector_size is %f \n" %float(shell_sector_size))

text_file.write("\n########################################\n")

text_file.write("\nSample sim info is \n%s \n" %str(simulation.info))

text_file.write("\n########################################")

######################################################################################################################

text_file.close()


# ## Run the actual simulation

# #### Define TMAX

# In[50]:

AGM_t_max = int(np.load('T_max.npy'))

#print('AGM_t_max is', AGM_t_max)
#
#print()


# In[51]:


#np.save('AGM_t_max.npy', AGM_t_max)


# ### Define the trackers

# #### Tracking interval

# In[52]:

#
#print('tracking_interval is', tracking_interval)
#
#print()


desired_timepoints = int(1e3)

tracking_interval =  int(AGM_t_max/desired_timepoints) # Frequency for writing out the data

# ### Droplet Tracker

# In[53]:


droplet_tracker = agent_based.DropletAgentTracker(interval = tracking_interval,
                                                  store_droplet_tracks = True,
                                                  store_emulsions = True)


# ### Background Tracker

# In[54]:


# storage = pde.FileStorage('AGM_background.hdf5')


# In[55]:


# background_storage_tracker = agent_based.BackgroundTracker(storage.tracker(tracking_interval))


# ### Run the simulation

# In[56]:
#
#print('default_timestep_based_on_background is', default_timestep_based_on_background)
#
#print()
#
default_timestep = simulation.estimate_dt(background_plus_agents)
#
#print('default_timestep is', default_timestep)
#
#print()
#
aggressive_default_timestep = 0.01*simulation.estimate_dt(background_plus_agents)

#print('aggressive_default_timestep is', aggressive_default_timestep)
#
#print()
#
#print('max_timestep is', max(default_timestep_based_on_background, default_timestep, aggressive_default_timestep))
#
#print()
#
#experimental_timestep = 10*max(default_timestep_based_on_background, default_timestep, aggressive_default_timestep)
#
#print('experimental_timestep is', experimental_timestep)
#
#print()

# In[57]:


# result = simulation.run(background_plus_agents, t_range = AGM_t_max,
#                         tracker = ['progress',
#                                    droplet_tracker,
#                                    background_storage_tracker])

########################################################################

result = simulation.run(background_plus_agents, t_range = AGM_t_max, dt = aggressive_default_timestep,
                        tracker = ['progress', droplet_tracker])

print()


#final_background_structure = result.background.data

#filename_concentration_profile = str('Concentration_profile_background_') + str('X=') + str(round(AGM_grid.discretization[0], 1)) + ',' + str('S=') + str(round(shell_sector_size, 1)) + ',' + str('L=') + str(round(shell_thickness, 1)) + '.npy'
#
#np.save(filename_concentration_profile, final_background_structure)

# ### Plot the final background field

# In[58]:


# initial_background.plot(title = 'Initial State', vmin = C_bottom_y_AGM, vmax = C_top_y_AGM, filename = 'AGM_Initial_State.png', action = 'close')
#
# result.background.plot(title = 'Final State', vmin = C_bottom_y_AGM, vmax = C_top_y_AGM, filename = 'AGM_Final_State.png', action = 'close')


# ### Make a movie

# In[59]:


# visualization.movie_scalar(background_storage, 'AGM_2D_PassiveEmulsion.mov')


# ### Final droplet stats

# In[60]:


#result.agents.droplets.get_size_statistics()


# In[61]:


#if (result.agents.droplets.get_size_statistics()['radius_mean'] == 0):
#
#    raise ValueError('All droplets have disappeared')


# ### Save the droplet tracks and emulsions to a file

# In[62]:

# AGM_droplet_tracks_filename = str('AGM_droplet_tracks_') + str('X=') + str(round(AGM_grid.discretization[0], 1)) + ',' + str('S=') + str(round(shell_sector_size, 1)) + ',' + str('L=') + str(round(shell_thickness, 1)) + '.hdf5'
#
# droplet_tracker.droplet_tracks.to_file(AGM_droplet_tracks_filename)
#
#
# # In[59]:
#
#
# AGM_emulsions_filename = str('AGM_emulsions_') + str('X=') + str(round(AGM_grid.discretization[0], 1)) + ',' + str('S=') + str(round(shell_sector_size, 1)) + ',' + str('L=') + str(round(shell_thickness, 1)) + '.hdf5'
#
# droplet_tracker.droplet_tracks.to_file(AGM_emulsions_filename)

# ######################################################################################################################

# ## Extract the radii and postitions of the droplets

# ### Identify global droplet recording timestamps

# In[64]:


global_droplet_timestamps = np.asarray(droplet_tracker.emulsions.times)


# ####################################################################################################################

# ### Isolate position and radius of the droplet

# In[65]:


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


# In[66]:


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


# In[67]:


#np.save('R_simulations_AGM.npy', single_droplet_radius_with_zeros[0][-1])

#print('Initial droplet radius was:',single_droplet_radius_with_zeros[0][0],
#      'and final droplet radius is:', round(single_droplet_radius_with_zeros[0][-1], 2))
#
#print()


# ### Plot the final droplet_tracks

# In[81]:

for droplet in range(N_DROPLETS):

    plt.plot(global_droplet_timestamps, single_droplet_radius_with_zeros[droplet], '--k', linewidth = 3, label = 'AGM')

CH_radius = np.load('CH_droplet_radius.npy')[0]

CH_time = np.load('CH_droplet_times.npy')

plt.plot(CH_time, CH_radius, 'steelblue', linewidth = 3, label = 'CH')

#########################################################

plt.xlabel(r'$t$')

plt.ylabel(r'$\frac{R}{w}$')

#########################################################

#plt.minorticks_on()
#plt.grid(b = True, which = 'both', linewidth = 0.05)

plt.xlim(0, AGM_t_max)

# plt.ylim(initial_radius, CH_radius[-1])

plt.legend()

image_name = str('AGM_Radius_') + str('X=') + str(round(AGM_grid.discretization[0], 2)) +             ',' + str('S=') + str(round(shell_sector_size, 2)) +             ',' + str('L=') + str(round(shell_thickness, 2)) + '.png'

plt.title(r'$L=$' + str(int(AGM_SYSTEM_SIZE)) + ', $\Delta x=$' + str(round(AGM_grid.discretization[0], 1)) + 'w, ' +
                                    r'$\Delta s=$' + str(round(shell_sector_size, 1)) + 'w, ' +
                                    r'$\ell$=' + str(round(shell_thickness, 1)) + 'w')

# plt.savefig(image_name, dpi = 400, bbox_inches = 'tight')

plt.close()

# ### Plot location

# In[69]:


for droplet in range(N_DROPLETS):

#     plt.plot(global_droplet_timestamps, single_droplet_x_location[droplet], linewidth = 3,
#             label = 'AGM x-axis')

    plt.plot(global_droplet_timestamps, single_droplet_y_location[droplet],
             '--k', linewidth = 3, label = 'AGM: Y-axis')

#     plt.plot(global_droplet_timestamps, single_droplet_z_location[droplet], linewidth = 3,
#             label = 'AGM z-axis')

########################################################################
# #
CH_Z_location = np.load('CH_droplet_z_location.npy')[0]

file = open("limit_r.out", "r")
limit_r = int(file.read())

plt.plot(CH_time, CH_Z_location, linewidth = 3, label = 'CH: z-axis')

#########################################################

plt.xlabel(r'$t$')

plt.ylabel('Location/w')

#########################################################

#plt.minorticks_on()
#plt.grid(b = True, which = 'both', linewidth = 0.05)

#########################################################

plt.xlim(0, AGM_t_max)

# plt.ylim(0, CH_Z_location[-1] - limit_r/2)

plt.legend()

#########################################################

image_name = str('AGM_Location_') + str('X=') + str(round(AGM_grid.discretization[0], 2)) +             ',' + str('S=') + str(round(shell_sector_size, 2)) +             ',' + str('L=') + str(round(shell_thickness, 2)) + '.png'

plt.title(r'$L=$' + str(int(AGM_SYSTEM_SIZE)) + ', $\Delta x=$' + str(round(AGM_grid.discretization[0], 1)) + 'w, ' +
                                    r'$\Delta s=$' + str(round(shell_sector_size, 1)) + 'w, ' +
                                    r'$\ell$=' + str(round(shell_thickness, 1)) + 'w')

# plt.savefig(image_name, dpi = 400, bbox_inches = 'tight')

plt.close()



# ### Save all arrays

# In[70]:


# np.save('AGM_single_droplet_times.npy', single_droplet_timestamps)

# np.save('AGM_droplet_times.npy', global_droplet_timestamps)

####################################################################

# np.save('AGM_global_droplet_radius_original.npy', single_droplet_radius)
#
# np.save('AGM_droplet_radius.npy', single_droplet_radius_with_zeros)
#
# ####################################################################
#
# np.save('AGM_droplet_x_location.npy', single_droplet_x_location)
#
# np.save('AGM_droplet_y_location.npy', single_droplet_y_location)
#
# np.save('AGM_droplet_z_location.npy', single_droplet_z_location)


# In[78]:


filename_T = str('Time_') + str('X=') + str(round(AGM_grid.discretization[0], 1)) + ',' + str('S=') + str(round(shell_sector_size, 1)) + ',' + str('L=') + str(round(shell_thickness, 1)) + '.npy'

filename_R = str('Radius_') + str('X=') + str(round(AGM_grid.discretization[0], 1)) + ',' + str('S=') + str(round(shell_sector_size, 1)) + ',' + str('L=') + str(round(shell_thickness, 1)) + '.npy'

filename_Y = str('Y_location_') + str('X=') + str(round(AGM_grid.discretization[0], 1)) + ',' + str('S=') + str(round(shell_sector_size, 1)) + ',' + str('L=') + str(round(shell_thickness, 1)) + '.npy'


# In[79]:


np.save(filename_T, global_droplet_timestamps)

np.save(filename_R, single_droplet_radius)

np.save(filename_Y, single_droplet_y_location)


## ### Number of drops with time
#
## In[73]:
#
#
#global_droplet_number = []
#
#for snapshot in range(len(global_droplet_timestamps)):
#
#    global_droplet_number.append(len(droplet_tracker.emulsions.emulsions[snapshot]))
#
#
## In[74]:
#
#
#np.save('AGM_droplet_number.npy', global_droplet_number)
#
#
## ### Plot Droplets vs Time
#
## In[75]:
#
#
#plt.plot(global_droplet_timestamps, global_droplet_number, linewidth = 3)
#
#plt.xlabel(r'$t$')
#
#plt.ylabel(r'$N_{drops}$')
#
#plt.minorticks_on()
#plt.grid(b = True, which='both', linewidth=0.05)
#
#plt.title(str(3) + 'D AGM Passive Droplet, ' + r'$L=$' + str(AGM_SYSTEM_SIZE) + ', ' +
#          str(N_DROPLETS) + ' Droplet, ' +
#          r'$\Delta x=$' + str(round(AGM_grid.discretization[0], 2)) + ', ' +
#          r'$\Delta s=$' + str(round(shell_sector_size, 2)) + ', ' +
#          r'$\ell$=' + str(round(shell_thickness, 2)))
#
#plt.xlim(0, AGM_t_max)
#
#plt.ylim(0, N_DROPLETS + 1)
#
## plt.savefig('AGM-Droplets vs Time', dpi = 400, bbox_inches = 'tight')
#
#plt.close()
#
#
## In[76]:
#
#
#print('N_Drops initial is', global_droplet_number[0], ', N_Drops final is', global_droplet_number[-1])
#
#print()

####################################################################################################################

# AGM_background_concentration_support_points = result.background.field.get_line_data()['data_x']

# AGM_background_concentration_profile_y = result.background.field.get_line_data()['data_y']

# AGM_background_concentration_profile_z = result.background.field.get_line_data()['data_z']

# filename_support_points = str('support_points_') + str('X=') + str(round(AGM_grid.discretization[0], 1)) + ',' + str('S=') + str(round(shell_sector_size, 1)) + ',' + str('L=') + str(round(shell_thickness, 1)) + '.npy'

# filename_concentration_profile_y = str('Concentration_profile_Y_') + str('X=') + str(round(AGM_grid.discretization[0], 1)) + ',' + str('S=') + str(round(shell_sector_size, 1)) + ',' + str('L=') + str(round(shell_thickness, 1)) + '.npy'

# filename_concentration_profile_z = str('Concentration_profile_Z_') + str('X=') + str(round(AGM_grid.discretization[0], 1)) + ',' + str('S=') + str(round(shell_sector_size, 1)) + ',' + str('L=') + str(round(shell_thickness, 1)) + '.npy'

# # In[69]:


# np.save(filename_support_points, AGM_background_concentration_support_points)

# np.save(filename_concentration_profile_y, AGM_background_concentration_profile_y)

# np.save(filename_concentration_profile_z, AGM_background_concentration_profile_z)

# ####################################################################################################################

# In[77]:

# import os
#
# final_string_1 = 'rm '+ AGM_droplet_tracks_filename
#
# os.system(final_string_1)
#
# final_string_2 = 'rm '+ AGM_emulsions_filename
#
# os.system(final_string_2)

####################################################################################################################
