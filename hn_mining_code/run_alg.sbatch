#!/bin/bash
#SBATCH --job-name=hn-mine
#SBATCH -N1                          # Ensure that all cores are on one machine
#SBATCH --partition=titanx-long              # Partition to submit to (serial_requeue)
#SBATCH --mem=5000                     # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=logs/result_%A_%a.out            # File to which STDOUT will be written
#SBATCH --error=logs/result_%A_%a.err            # File to which STDERR will be written
#SBATCH --gres=gpu:1
#SBATCH --array=19
## Usage:sbatch
## # sbatch scripts/run_video_cluster. ${VIDEO_PATH} ${DATASET_NAME}

python get_hardNegative.py --videoPath train_detection_videos/video${SLURM_ARRAY_TASK_ID} --detectionsFilePath train_detection_outputs/video${SLURM_ARRAY_TASK_ID}.txt  --validScore 0.8 --alpha 5 --validSimScore 0.5 --iouThr 0.2 --beta 100
