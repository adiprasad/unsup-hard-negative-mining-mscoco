"""
Code to put bounding boxes on detections in a video by reading the detections file.

Takes a video and its detections text file and draws bounding boxes over all the detections(true positives and hard negatives) in that video. 
The code allows slowing down the video before and after every hard negative and freezing at the hard negative frames for configurable(supplied as parameters) amount of time. 

    Author: Aditya Prasad
    Date Created: 02/20/2018
"""

import numpy as np 
import cv2
import os
from collections import deque


class VideoMaker(object):

	'''
	mode : face-detection : 0, object_detection = 1
	path_to_video : path to the input video
	path_to_bbox_file : path to the bounding box file
	path_to_fp_frames_file : output path of the file containing false positive frames only
	num_classes : number of classes
	duration_bef : amount of seconds before the false positive frame for which the video 
				   should run in slow down mode
	duration_after : amount of seconds after the false positive frame for which the video should 
				    run in slow down mode
	speed_around_fp : Speed of the video in the [-duration_bef, duration_after] interval 
					relative to the original speed
	'''
	def __init__(self, mode, path_to_video, path_to_bbox_file, path_to_fp_frames_file, num_classes,
				output_video_path, duration_bef=3, duration_after=3, speed_around_fp = 0.5, pause_at_fp = 2):
		
		self._mode = mode
		self._path_to_video = path_to_video
		self._output_video_path = output_video_path
		self._path_to_bbox_file = path_to_bbox_file
		self._path_to_fp_frames_file = path_to_fp_frames_file
		self._num_classes = num_classes
		self._duration_bef = duration_bef
		self._duration_after = duration_after
		self._speed_around_fp = speed_around_fp
		self._duplicacy = int(round(1/float(speed_around_fp))) 		# How many times should a frame be duplicated in slow mode
		self._pause_at_fp = pause_at_fp

		self.__load_video()
		self.__load_bbox_file()
		self.__init_fp_frames_file()
		self.__init_queue()


	'''
	Make the new video of a specified portion with bounding boxes

	Input args :-
	start_time : starting time coordinate in the original video
	end_time : ending time coordinate in the original video

	'''
	def make_bbox_video(self, start_time=0, end_time='eov'):
		start_frame, end_frame = self.__get_start_end_frame_numbers(start_time, end_time)

		print("Start frame : {0}".format(start_frame))
		print("End frame : {0}".format(end_frame))

		self.__init_out_video()

		if self._mode == 0:
			self.__face_detection_video(start_frame, end_frame)
		else:
			self.__object_detection_video(start_frame, end_frame)

		self.__close_all()


	def __face_detection_video(self, start_frame, end_frame):
		frame_counter = 0 

		self.__fill_queue_with_frames()
		frame_counter+=self._q_size

		while (frame_counter < end_frame):
			self.__append_frame_to_output()
			processed_frame, fp_frame = self.__get_next_frame()
			self.__add_frame_to_queue(processed_frame, fp_frame)
			frame_counter+=1

		# Write the frames remaining in the queue to the video
		while (len(self._frame_q) > 0):
			self.__append_frame_to_output()
		
	def __append_frame_to_output(self):
		if self._last_fp_in_queue >= self._left_vicinity :				# If the last false positive is within (-duration_bef, duration_after) of the current frame
			self.__write_current_frame_to_output(self._duplicacy)
		else:
			self.__write_current_frame_to_output(1)

	def __write_current_frame_to_output(self, duplicacy):
		current_frame, fp_frame = self._frame_q.popleft()

		for i in range(duplicacy):
			self._out.write(current_frame)

		# Special pause at FP frame for self._pause_at_fp seconds
		if fp_frame == True:
			pause_seconds = int(self._fr * self._pause_at_fp)
			
			for i in range(pause_seconds):
				self._out.write(current_frame)


	'''
	Initialize the deque with as many frames as it should take, depending on duration_bef and duration_after
	'''
	def __fill_queue_with_frames(self):

		for i in range(self._q_size):
			processed_frame, fp_frame = self.__get_next_frame()
			self.__add_frame_to_queue(processed_frame, fp_frame)


	def __add_frame_to_queue(self, frame, fp_frame):
		self._frame_q.append((frame, fp_frame))

		if fp_frame == True:
			self._last_fp_in_queue = self._q_size
		
		self._last_fp_in_queue -= 1 			# Reducing the position of last seen false positive frame in the queue


	'''
	Reads in, processes the incoming frame and returns it back
	'''
	def __get_next_frame(self):
		frame_number = self.__get_frame_number_from_text()
		frame_img = self.__get_frame_from_video()
		frame_img_processed, fp_frame = self.__process_frame(frame_img, frame_number)
		
		return frame_img_processed, fp_frame

		
	def __get_frame_number_from_text(self):
		line = self._bbox_file.readline()
		line = line.strip()

		comma_sep_array = line.split(',')
		frame_number = int((comma_sep_array[0].split(':'))[1])

		return frame_number


	'''
	Read line to get the number of boxes
	'''
	def __get_number_of_boxes_from_text(self):
		line = self._bbox_file.readline()
		line = line.strip()

		box_num = int(line)

		return box_num


	'''
	Get an array of bounding box coordinates and fp status
	'''
	def __get_bbox_coordinates(self, box_num):
		#  List of bbox detections of the type [(left,top),(right,bottom),cls,fp]
		bbox_coords_list  = []

		for i in range(box_num):
			line = self._bbox_file.readline()
			line = line.strip()

			coords_cls_fp = line.split(" ")
			left_top, right_bottom = self.__get_rectangle_coordinates(coords_cls_fp)
			cls = coords_cls_fp[4]
			fp = coords_cls_fp[5]

			coords = [left_top, right_bottom, cls, fp]

			bbox_coords_list.append(coords)

		return bbox_coords_list

	def __get_rectangle_coordinates(self, coords_cls_fp):
		left = coords_cls_fp[0]
		top = coords_cls_fp[1]
		width = coords_cls_fp[2]
		height = coords_cls_fp[3]

		left = int(float(left))
		top = int(float(top))
		width = int(float(width))
		height = int(float(height))

		right = left+width
		bottom = top+height

		return (left, top), (right, bottom)

	def __get_frame_from_video(self):
		ret, frame = self._cap.read()

		if ret == False:
			print "Could not read Frame!!"

		return frame


	def __put_box_on_frame(self, frame_img, box_coords):
		left_top, right_bottom, cls, fp_frame = box_coords[0], box_coords[1], box_coords[2], box_coords[3]

		fp_frame = int(fp_frame)

		if fp_frame > 0:
			boxColor = (0,255,0)
		else:
			boxColor = (0, 0, 255)

		frame_img = cv2.rectangle(frame_img,left_top,right_bottom,boxColor,3) # To show the rectangle around the detections
		frame_img = cv2.putText(frame_img , cls, (left_top[0],right_bottom[1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)

		return frame_img, fp_frame


	def __put_fnum_time_on_frame(self, frame_img, frame_num):
		height, width = frame_img.shape[0], frame_img.shape[1]
		#frame_img = cv2.putText(frame_img ,str(frame_num), (10, height - 50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
		frame_time = self.__get_time_from_frame_number(frame_num)
		frame_img = cv2.putText(frame_img ,str(frame_num) + "," + frame_time, (7, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
		
		return frame_img
		

	'''
	Put bounding boxes and text boxes on the frame
	'''
	def __process_frame(self, frame_img, frame_number):
		box_num = self.__get_number_of_boxes_from_text()
		box_coords_list = self.__get_bbox_coordinates(box_num)
		fp_sum_flag = 0

		for box_coords in box_coords_list:
			frame_img, fp_frame = self.__put_box_on_frame(frame_img, box_coords)
			fp_sum_flag += (1 - fp_frame)

		frame_img = self.__put_fnum_time_on_frame(frame_img, frame_number)
		is_frame_fp = fp_sum_flag>0

		self.__write_framenum_to_fp_frames_file(frame_number, is_frame_fp)

		print("Processed Frame number {0}".format(frame_number))

		return frame_img, is_frame_fp


	def __object_detection_video(self, start_frame, end_frame):
		pass	



	def __init_out_video(self):
		#fourcc = cv2.VideoWriter_fourcc(self._cap.get(6))
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		self._out = cv2.VideoWriter(self._output_video_path, fourcc, self._fr, self._res)


	'''
	Convert the start_time and end_time markers into start_frame and end_frame numbers
	'''
	def __get_start_end_frame_numbers(self, start_time, end_time):
		start_frame = 0 
		end_frame = int(round(self._cap.get(7)))

		if start_time != 0:
			start_time = self.__calculate_time_in_seconds(start_time)
			start_frame = self._fr * start_time

		if end_time != 'eov':
			end_time = self.__calculate_time_in_seconds(end_time)
			end_frame = self._fr * end_time

		return start_frame, end_frame


	'''
	Input :-
	time : Format HH:MM:SS 

	returns time converted in seconds
	'''
	def __calculate_time_in_seconds(self, time):
		t_arr = time.split(':')
		t_sec = int(t_arr[0])*3600 + int(t_arr[1])*60 + int(t_arr[2])

		return t_sec


	def __get_time_from_frame_number(self, frame_number):
		seconds_elapsed = int(round(frame_number/float(self._fr)))

		hours_elapsed = seconds_elapsed/3600
		minutes_left_in_seconds = seconds_elapsed - 3600*hours_elapsed
		minutes_elapsed = minutes_left_in_seconds/60
		seconds_elapsed = minutes_left_in_seconds - 60*minutes_elapsed

		hh_mm_ss = "{0}:{1}:{2}".format(hours_elapsed, minutes_elapsed, seconds_elapsed)

		return hh_mm_ss


	'''
	Initialize a queue of frames with a size depending on duration_bef, duration_after
	and video frame rate. The idea is that the frames in the new video will  be stacked 
	at the original speed if there is no hard negative frame in the queue. Otherwise,
	the frames will be stacked in such a way that speed_around_fp is maintained in the
	video from [-duration_bef, duration_after] around every hard negative
	'''
	def __init_queue(self):
		self._q_size = self.__return_queue_size()
		self._frame_q = deque(maxlen = self._q_size)

	def __return_queue_size(self):
		return int(self._duration_bef*self._fr*self._speed_around_fp)
		
	def __close_all(self):
		self._out.release()
		self._cap.release()
		self._bbox_file.close()
		self._fp_frames_file.close()

		cv2.destroyAllWindows()

	'''
	Loads the video into a VideoCapture object
	'''
	def __load_video(self):
		self._cap = cv2.VideoCapture(self._path_to_video)

		if (self._cap.isOpened()):
			self.__get_frame_rate()
			self.__get_native_resolution()

		print ".......Video ready for processing......."


	'''
	Gets the frame rate of the video
	'''
	def __get_frame_rate(self):
		self._fr = int(round(self._cap.get(5)))

	'''
	Gets the native resolution of the video
	'''
	def __get_native_resolution(self):
		width = self._cap.get(3)
		height = self._cap.get(4)

		self._res = (int(width), int(height))

	'''
	Load the bbox file for reading
	'''
	def __load_bbox_file(self):
		self._bbox_file = open(self._path_to_bbox_file, 'r')
		self._left_vicinity = - int(self._fr * self._duration_after * self._speed_around_fp)			# left_vicinity defines the number of frames for which the video has to run in slow mo, even after a false positive has passed
		self._last_fp_in_queue = self._left_vicinity - 1 		# Initial value of the position of the most recent false positive in the queue, wrt the frame at position 0 in the queue

		print ".......Bounding Box file found for processing......"


	def __init_fp_frames_file(self):
		self._fp_frames_file = open(self._path_to_fp_frames_file, 'a+')

	def __write_framenum_to_fp_frames_file(self, frame_number, is_frame_fp):
		if is_frame_fp == True:
			self._fp_frames_file.write(str(frame_number) + "\n")





		