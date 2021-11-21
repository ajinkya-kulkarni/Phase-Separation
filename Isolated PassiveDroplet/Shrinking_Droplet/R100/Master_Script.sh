############# BEGIN CLUSTER STUFF ##############

#!/bin/bash

#$ -S /bin/bash

# Preserve environment variables
#$ -V

# Execute from current working directory
#$ -cwd

#################### END OF CLUSTER STUFF #######################


clear

unset all

rm One_Drop_*

echo ""

rm Time_*.npy

echo ""

rm Radius_*.npy

echo ""

rm AGM_*.npy

echo ""

rm AGM_*.png

echo ""

rm AGM_*.hdf5

echo ""

rm AGM_3D_edited_*

echo ""

#################################################################

# AGM system size = 1600

# support_points_GRID_list=(200) # All even numbers

support_points_GRID_list=(200 160 100 80 50 40 20 10 8 4 2) # All even numbers

shell_thickness_list=(8 10 16 20 32 40 50 80 100 160 200 400)

shell_sector_size_list=("${shell_thickness_list[@]}")

# grid_discretization_list=(100 200)
# for i in "${grid_discretization_list[@]}"; do
#   echo "$i";
# done

################################################################

COUNTER=0

for i in "${support_points_GRID_list[@]}"; do

  echo ""

  echo ""

  sleep 10m

  echo ""

  echo ""

    for j in "${shell_thickness_list[@]}"; do

        for k in "${shell_sector_size_list[@]}"; do

            sed -e "s/CHANGE_support_points_GRID/$i/g" -e "s/CHANGE_shell_thickness/$j/g" -e "s/CHANGE_shell_sector_size/$k/g" AGM_3D_MASTER_Parallel.py > AGM_3D_edited_Parallel_N$i,L$j,s$k.py

            ############################################################

            qsub -q teutates.q -N One_Drop_N$i AGM_3D_edited_Parallel_N$i,L$j,s$k.py
            #
            # python3 AGM_3D_edited_Parallel_N$i,L$j,s$k.py
            #
            # rm AGM_3D_edited_Parallel_N$i,L$j,s$k.py

            ############################################################

#             echo Current counter is $COUNTER out of $LAST_COUNTER

            ############################################################

            COUNTER=$((COUNTER+1))

            ############################################################

        done # endloop for k

    done # endloop for j

done # endloop for i

echo ""

clear
