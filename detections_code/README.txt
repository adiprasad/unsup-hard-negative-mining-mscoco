To use the detections code, download all your videos inside a parent folder say 'downloaded_videos' with the following structure :-

downloaded_videos
|
|--video1
|	|--video1.mkv
|
|--video2
|	|--video2.mkv
|	
..
..
..

Helper code is available to convert the videos present inside a folder to the above mentioned folder structure. 

Steps :-

1. Specify the path of model weights to be used for running the network on the videos on line 28.
2. Decide an output_folder where the detection txt files will be placed.
3. Decide a confidence_threshold crossing which the detections will be reported.  

3. Usage :-

python2 tools/test_dets_dog_detector.py \
--downloaded_videos $1 \
--video ID \
--out_folder output_folder \
--conf_thresh confidence_threshold


where ID : 1,2,3 etc. as per the videos have been saved inside the downloaded_videos folder hierarchy