
-------------------------------------------------------------
DOCUMENTATION FOR THE VIDEO MAKER CODE
-------------------------------------------------------------

The VideoMaker converts an original video to a video with bounding boxes around the detections. There converted video can be configured to slow down around false positives at a speed of the slowdown relative to the speed of original video for ex 0.2x. 

How many seconds before and after a false positive should the video be slowed down and the slow down speed(0.2x) are configurable parameters. One can also set the number of seconds they want the video to freeze at each false positive. Also, if you don't want to convert the entire video, you can specify the start and end times in HH-MM-SS format and the converted video will be created for that sub-duration only.

The outputs are :-

1) Converted video with bounding boxes and frame number information
2) Text file containing the frame numbers which contain false positives.

------------------------------------------------------------------------------------------

The VideoMaker class takes in following arguments on initialization :-

mode : 0 for single class detection, 1 for multiclass detection (Mode 0 available as of 02/20/2018)
path_to_video : Path to the original video
path_to_bbox_file : Path to the txt file containing bounding boxes
path_to_fp_frames_file : Output path to a txt file that will contain the frame numbers of the frames containing false positives
num_classes : auxillary argument (to be specified in mode 1)
output_video_path : Output path to the converted video (Must end in .avi)
duration_bef : The number of seconds before a false positive the video should start slowing down
duration_after : The number of seconds after a false positive the video should run in slowdown mode 
speed_around_fp : Speed of slowdown(relative to the original video speed) around false postive. This should be a decimal value like 0.2 for 0.2x
pause_at_fp : Number of seconds the video should freeze at the false positive frame


------------------------------------------------------------------------------------------


Steps to follow :-

1) from videomaker import VideoMaker
2) Decide the arguments and initialize the VideoMaker

VM = VideoMaker(args)

3) VM.make_bbox_video(start_time=HH:MM:SS, end_time=HH:MM:SS)

* If you want to convert the entire video, do not pass the start_time and end_time args.
* If you want to convert from a start time to the end of the video, pass the start_time argument only.

------------------------------------------------------------------------------------------