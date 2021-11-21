
############# BEGIN CLUSTER STUFF ##############

#!/bin/bash

#$ -S /bin/bash

# Preserve environment variables
#$ -V

# Execute from current working directory
#$ -cwd

#################### END OF CLUSTER STUFF #######################


clear

cd ./R20

rm One_Drop_*

rm AGM_3D_edited_*

rm AGM_*.hdf5

rm -rf Data

mkdir Data

mv Time_* ./Data

mv Radius_* ./Data

cd ..

wait

cd ./R40

rm One_Drop_*

rm AGM_3D_edited_*

rm AGM_*.hdf5

rm -rf Data

mkdir Data

mv Time_* ./Data

mv Radius_* ./Data

cd ..

wait

cd ./R80

rm One_Drop_*

rm AGM_3D_edited_*

rm AGM_*.hdf5

rm -rf Data

mkdir Data

mv Time_* ./Data

mv Radius_* ./Data

cd ..

wait


cd ./R100

rm One_Drop_*

rm AGM_3D_edited_*

rm AGM_*.hdf5

rm -rf Data

mkdir Data

mv Time_* ./Data

mv Radius_* ./Data

cd ..

wait


cd ./R160

rm One_Drop_*

rm AGM_3D_edited_*

rm AGM_*.hdf5

rm -rf Data

mkdir Data

mv Time_* ./Data

mv Radius_* ./Data

cd ..

wait


cd ./R200

rm One_Drop_*

rm AGM_3D_edited_*

rm AGM_*.hdf5

rm -rf Data

mkdir Data

mv Time_* ./Data

mv Radius_* ./Data

cd ..

wait

clear
