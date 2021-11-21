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

rm Pair*

echo ""

rm Time_*.npy

echo ""

rm MeanRadius_*.npy

echo ""

rm AGM_3D_edited_*

echo ""

rm MeanRadius_*.png

echo ""

rm AGM_*.hdf5

echo ""

#################################################################

# AGM system size = 1200

# support_points_GRID_list=(3 6)

support_points_GRID_list=(3 6 15 30 60 120 240)

#################################################################

# shell_thickness_list=(2 5)

shell_thickness_list=(2 5 10 20 40 50 70 160 200 400)

# shell_sector_size_list=(5 10)

shell_sector_size_list=(5 10 20 40 70)

# grid_discretization_list=(100 200)
# for i in "${grid_discretization_list[@]}"; do
#   echo "$i";
# done

# shell_sector_size_list=("${shell_thickness_list[@]}")

################################################################

COUNTER=0

for i in "${support_points_GRID_list[@]}"; do

  echo ""

  echo ""

  echo "Current support points are $i"

  sleep 5m

  echo ""

  echo ""

    for j in "${shell_thickness_list[@]}"; do

        for k in "${shell_sector_size_list[@]}"; do

            sed -e "s/CHANGE_support_points_GRID/$i/g" -e "s/CHANGE_shell_thickness/$j/g" -e "s/CHANGE_shell_sector_size/$k/g" AGM_3D_MASTER_Parallel.py > AGM_3D_edited_Parallel_N$i,L$j,s$k.py

            ############################################################

           qsub -q teutates.q -N Pair_N$i AGM_3D_edited_Parallel_N$i,L$j,s$k.py

            # python3 AGM_3D_edited_Parallel_N$i,L$j,s$k.py
            #
            # rm AGM_3D_edited_Parallel_N$i,L$j,s$k.py

            ############################################################

            COUNTER=$((COUNTER+1))

            ############################################################

        done # endloop for k

    done # endloop for j

done # endloop for i

echo ""
