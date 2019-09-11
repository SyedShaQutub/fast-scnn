# ==============================================================================
#
# This script is used to run fast-SCNN on Kitti Dataset. Users could also
# modify from this script for their use case.
#


# Exit immediately if a command exits with a non-zero status.
set -e

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`
export HDF5_USE_FILE_LOCKING=FALSE

CURRENT_DIR=$(pwd)

# Run model_test first to make sure the Tensorflow-gpu=2.0 is correctly set.
python "${CURRENT_DIR}"/model_test.py -v

DATASET_DIR="dataset"
cd "${CURRENT_DIR}/dataset"

DATA_DIR="${CURRENT_DIR}/${DATASET_DIR}"
KITTIDATA_DIR="${DATA_DIR}/kitti_seg"
PROCDATA_DIR="${DATA_DIR}/kitti_seg/data_semantics/proc_data"
# Go to datasets folder and download Kitti segmentation dataset.
sh download_kitti.sh
 
cd ../

TRAIN_SPLIT=0.85
VAL_SPLIT=0.15
BATCH_SIZE=3
NO_OF_EPOCHS=1000
NUM_CLASSES=34

echo "Converting KITTI dataset..."
python ./"${DATASET_DIR}"/prepare_data.py \
  --dataset_dir="${DATA_DIR}" \
  --train_split="${TRAIN_SPLIT}" \
  --batch_size="${BATCH_SIZE}" \
  --val_split="${VAL_SPLIT}"

python "${CURRENT_DIR}"/train_keras.py \
  --procdata_dir="${PROCDATA_DIR}" \
  --kittidata_dir="${KITTIDATA_DIR}" \
  --no_of_epochs="${NO_OF_EPOCHS}" \
  --num_classes="${NUM_CLASSES}" \
  --batch_size="${BATCH_SIZE}"



