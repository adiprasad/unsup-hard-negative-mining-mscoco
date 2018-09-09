#! /bin/bash

#source ~/.bashrc
#source ~/mypython/bin/activate
#cd /home/ashishsingh/


python get_hardNegative.py --videoPath train_detection_videos/video${1} --detectionsFilePath train_detection_outputs/video${1}.txt  --validScore 0.8 --alpha 5 --validSimScore 0.5 --iouThr 0.2 --beta 100

