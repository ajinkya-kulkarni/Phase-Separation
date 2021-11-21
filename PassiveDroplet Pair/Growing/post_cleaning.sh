
############# BEGIN CLUSTER STUFF ##############

#!/bin/bash

#$ -S /bin/bash

# Preserve environment variables
#$ -V

# Execute from current working directory
#$ -cwd

#################### END OF CLUSTER STUFF #######################


clear

rm -rf Data

mkdir Data

####################

mv Time_* ./Data

mv MeanRadius_* ./Data

####################

rm Pair_*

rm AGM_3D_edited_Parallel_*

rm AGM_*.hdf5

####################

clear
