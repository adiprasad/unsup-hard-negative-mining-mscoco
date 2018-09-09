"""
Code to edit COCO annotations file and augment it with supplementary images along with ground truth detections.
This enabled us re-training the detector on the mined hard negative frames.
    Author: Aditya Prasad
    Date Created: 02/20/2018
"""

import argparse
import json 
import numpy as np
import os 
from os.path import join, exists, dirname
import copy
import cv2
from pycocotools.coco import COCO
import shutil

class EditAnnotation(object):

	def __init__(self, mode_year, ann_source_path, ann_dest_path, category_id, hard_neg_dir, data_dir, dest_dir):
		self.mode_year = mode_year
		self.ann_source_path = ann_source_path
		self.ann_dest_path = ann_dest_path
		self.cat_id = category_id
		self.hn_dir = hard_neg_dir
		self.data_dir = data_dir
		self.dest_dir = dest_dir

		self.__move_old_files_to_dest()


	def replace_annotation_file(self):
		ann_json = self.__edit_annotations()
		ann_dest_file_path = self.__return_dest_of_annotation_file()

		print "Annotation edit completed"

		with open(ann_dest_file_path, "w") as f:
			json.dump(ann_json, f)

		print "New annotation file dumped"


	'''
	Loads and returns the current annotations file based on the mode_year parameter
	'''
	def __load_annotation_json(self):
		ann_file_path = self.__return_path_of_annotation_file()
		with open(ann_file_path, 'r') as f:
			ann_json = json.load(f)

		return ann_json


	def __return_path_of_annotation_file(self):
		return self.ann_source_path

	def __return_dest_of_annotation_file(self):
		return self.ann_dest_path


	def __return_path_of_weights_folder(self):
		data_dir_arr = self.data_dir.split("/")
		parent_dir = "/".join(data_dir_arr[0:-1])

		weights_dir  = join(parent_dir, 'output')
		
		return weights_dir


	def __move_old_files_to_dest(self):
		iter_dest_dict_path = self.__get_iter_dest_path()

		self.__move_cache_to_dest(iter_dest_dict_path)
		self.__copy_annotations_to_dest(iter_dest_dict_path)
		self.__copy_weights_to_dest(iter_dest_dict_path)


	def __get_iter_dest_path(self):
		if not os.path.exists(self.dest_dir):
			os.makedirs(self.dest_dir)
			curr_iter_num = "0"
		else:
			dirs = os.listdir(self.dest_dir)
			dir_ints = [int(x) for x in dirs]  		# dest dir should contain only iter number directories, 0 is for original COCO
			curr_iter_num = str(max(dir_ints) + 1)

		iter_dest_dict_path = join(self.dest_dir, curr_iter_num)
		
		#os.makedirs(iter_dest_dict_path)

		return iter_dest_dict_path


	def __move_cache_to_dest(self, dest_path):
		cache_path = join(self.data_dir, "cache")

		print "Copying old cache to iterations folder"

		shutil.copytree(cache_path, dest_path, symlinks=True)

		print "Deleting old cache"

		shutil.rmtree(cache_path)


	def __copy_annotations_to_dest(self, dest_path):
		ann_file_path = self.__return_path_of_annotation_file()

		print "Copying old annotation to iterations folder"

		shutil.copy(ann_file_path, dest_path)

	def __copy_weights_to_dest(self, dest_path):
		weights_dir = self.__return_path_of_weights_folder()

		print "Copying old weights to iterations folder"

		shutil.copytree(weights_dir, join(dest_path,'output'), symlinks=True)

	'''
	Returns the list of files in the Hard negatives folder
	'''
	def __return_list_of_files_in_hn(self):
		dirnames = os.listdir(self.hn_dir)
		filenames = filter(lambda x : not x.endswith('txt'), dirnames)

		return filenames

	'''
	Loads the annotation file, edits it by adding all the pseudo-ground truths 
	Moves the frames to the coco dataset folder
	'''
	def __edit_annotations(self):
		ann_json = self.__load_annotation_json()
		
		frame_bbox_dict = self.__structurize_hard_negatives()

		print len(frame_bbox_dict)

		self.__put_annotations_and_move_images(ann_json, frame_bbox_dict)

		return ann_json

	'''
	Puts the annotations into the annotations file and
	moves the image to the coco directory
	'''
	def __put_annotations_and_move_images(self, ann_json, frame_bbox_dict):
		max_img_id, max_ann_id = self.__get_max_img_id_ann_id()
		self._img_id = max_img_id
		self._ann_id = max_ann_id


		for img, bbox_arr in frame_bbox_dict.iteritems():
			img_name_for_coco = self.__put_annotations(ann_json, img, bbox_arr)
			self.__move_image_to_coco_dir(img, img_name_for_coco)


	def __get_next_img_id(self):
		self._img_id+=1

		return self._img_id

	def __get_next_ann_id(self):
		self._ann_id+=1

		return self._ann_id

	def __get_max_img_id_ann_id(self):
		ann_file_path = self.__return_path_of_annotation_file()
		coco = COCO(ann_file_path)

		img_ids = coco.getImgIds()
		ann_ids = coco.getAnnIds(imgIds=img_ids)

		return max(img_ids), max(ann_ids)

	
	def __put_annotations(self, ann_json, img_path, bbox_arr):
		img_ann = self.__put_image_annotation(ann_json, img_path)
		self.__put_bbox_annotations(ann_json, img_ann, bbox_arr)

		return img_ann["file_name"]


	def __put_image_annotation(self, ann_json, img_path):
		img_full_path = join(self.hn_dir, img_path)

		print img_full_path

		height, width = self.__get_img_height_width(img_full_path)

		img_ann = {}
		img_ann["width"] = width
		img_ann["height"] = height
		img_ann["id"] = self.__get_next_img_id()
		img_ann["file_name"] = self.__get_coco_filename(img_ann["id"])

		ann_json["images"].append(img_ann)

		return img_ann

	def __put_bbox_annotations(self, ann_json, img_ann, bbox_arr):
		for bbox in bbox_arr:
			ann_dict = {}
			
			if bbox[-1] == 1.0:		# Putting annotations for pseudo ground truths only
				ann_dict["segmentation"] = [np.random.normal(200, size = 20).tolist()]			# Filling seg with random numbers
				ann_dict["area"] = 0.5 * img_ann["width"] * img_ann["height"]						# Filling area with random number
				ann_dict["image_id"] = img_ann["id"]
				ann_dict["bbox"] = bbox[0:-1].tolist()
				ann_dict["category_id"] = self.cat_id
				ann_dict["id"] = self.__get_next_ann_id()
				ann_dict["iscrowd"] = 0

				ann_json["annotations"].append(ann_dict)

	def __move_image_to_coco_dir(self, img, img_name_for_coco):
		img_path = join(self.hn_dir, img)
		coco_dir = join(self.data_dir, "coco/images/{0}".format(self.mode_year))
		coco_file_path = join(coco_dir, img_name_for_coco)

		shutil.copyfile(img_path, coco_file_path)


	def __get_img_height_width(self, img_full_path):
		im = cv2.imread(img_full_path)

		return im.shape[0], im.shape[1]

	def __get_coco_filename(self, img_id):
		img_id = str(img_id)
		img_twlve_digit_id = img_id.zfill(12)
		coco_filename = "COCO_{0}_{1}.jpg".format(self.mode_year, img_twlve_digit_id)

		return coco_filename

	'''
	Structurizes all hard negative files
	'''
	def __structurize_hard_negatives(self):
		frame_bbox_dict = {}

		hn_files = self.__return_list_of_files_in_hn()
		
		for hn_file in hn_files:
			self.__read_hn_file(hn_file, frame_bbox_dict)

		frame_bbox_dict = self.__filter_frames_with_pseudo_truths(frame_bbox_dict)
		frame_bbox_dict = self.__throw_small_bboxes(frame_bbox_dict)

		print "Image to bounding box dict is ready....."

		return frame_bbox_dict

	'''
	Filters and returns frames and bbox information which have 
	at least one pseudo-ground truth
	'''
	def __filter_frames_with_pseudo_truths(self, frame_bbox_dict):
		filtered_dict = {k:v for k,v in frame_bbox_dict.iteritems() if not np.all(v[:,-1] == 0.0)}

		return filtered_dict

	def __throw_small_bboxes(self, frame_bbox_dict):
		filtered_frame_bbox_dict = {}

		for img, bbox_arr in frame_bbox_dict.iteritems():
			small_box_list = np.where(np.logical_and(bbox_arr[:,2] > 10, bbox_arr[:,3] > 10))[0]

			filtered_frame_bbox_dict[img] = bbox_arr[small_box_list]

		return filtered_frame_bbox_dict


	def __read_hn_file(self, file_name, frame_bbox_dict):
		fp  = self.__get_hn_file(file_name)

		while True:
			file_path = fp.readline().strip()
			
			if file_path == '':
				break

			num_bbox = int(fp.readline().strip())

			self.__put_bbox_to_dict(fp, frame_bbox_dict, num_bbox, file_path)


		self.__close_file(fp)


	'''
	Load the file and return the file pointer
	'''
	def __get_hn_file(self, file_name):
		file_path = join(self.hn_dir, file_name + ".txt")
		fp = open(file_path, "r")

		return fp

	def __put_bbox_to_dict(self, file_pointer, frame_bbox_dict, num_bbox, file_path):
		fp_array = file_path.split('/')
		rel_path = '/'.join(fp_array[1:])

		bbox_arr = np.empty((num_bbox, 5))

		for i in range(num_bbox):
			line = file_pointer.readline().strip()
			coords_label = line.split(" ")

			for c in range(5):
				bbox_arr[i][c] = float(coords_label[c])

		frame_bbox_dict[rel_path] = bbox_arr

		#print frame_bbox_dict[rel_path]


	def __close_file(self, fp):
		fp.close()


"""Parse input arguments."""
def parse_args():
    parser = argparse.ArgumentParser(description='Editing COCO annotation file')

    print "yay"
    
    parser.add_argument('--annFilePath', dest='annotation_file_path', help='Annotations file path')
    parser.add_argument('--annDestPath', dest='annotation_dest_path', help='New annotations file destination path')
    parser.add_argument('--modeYear', dest='mode_year', help='Mode and Year e.g. train2014', default='train2014')
    parser.add_argument('--hardNegDir', dest='hard_neg_dir', help='Hard Negatives Parent Dir path')
    parser.add_argument('--dataDir', dest='data_dir', help='Data directory of the repo')
    parser.add_argument('--destDir', dest='dest_dir', help='Data directory to move the old annotations and cache')
    parser.add_argument('--catId', dest='cat_id', help='Category ID', default=0, type=int)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
	args = parse_args()

	print args

	eann = EditAnnotation(args.mode_year, args.annotation_file_path, args.annotation_dest_path, args.cat_id, args.hard_neg_dir, args.data_dir, args.dest_dir)

	eann.replace_annotation_file()





