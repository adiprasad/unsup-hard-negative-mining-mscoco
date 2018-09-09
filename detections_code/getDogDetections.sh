#!/bin/bash
#SBATCH --job-name=detectron
#SBATCH -N1                          # Ensure that all cores are on one machine
#SBATCH --partition=m40-long              # Partition to submit to (serial_requeue)
#SBATCH --mem=50000                     # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=logs/result_%A_%a.out            # File to which STDOUT will be written
#SBATCH --error=logs/result_%A_%a.err            # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --array=1-15
## Usage:sbatch
## # sbatch scripts/run_video_cluster. ${VIDEO_PATH} ${DATASET_NAME}
echo `pwd`
echo $1
echo "SLURM task ID: "$SLURM_ARRAY_TASK_ID
##### Experiment settings #####
VIDEO_PATH=$1/video${SLURM_ARRAY_TASK_ID}       # argument to the script is the video name
OUTPUT_NAME=/mnt/nfs/scratch1/aprasad/dog_detection_outputs/${2}
MIN_SCORE=0.6
echo "Chunk path: "${VIDEO_PATH}

python2 tools/test_dets_dog_detector.py \
--video_folder $1 \
--video ${SLURM_ARRAY_TASK_ID} \
--out_folder ${OUTPUT_NAME} \
--conf_thresh $2

