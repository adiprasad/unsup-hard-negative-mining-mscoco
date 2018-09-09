import os 
from os.path import join
import pickle

parent_dir = os.getcwd()		# Set the parent folder of all the chunks here

mapping_txt_file = open(join(parent_dir, "video_name_key_map.txt"), "a+")

video_name_cntr = 1
filename_mapping_dict = {}


files_in_chunk = os.listdir(parent_dir)
mkv_files = filter(lambda x : x.endswith(".mkv"), files_in_chunk)

for file_name in mkv_files:
	file_path = join(parent_dir, file_name)

	os.mkdir(join(parent_dir,"video{0}".format(video_name_cntr)))
	new_video_path = join(parent_dir,"video{0}".format(video_name_cntr))

	new_file_path = join(new_video_path, "video{0}.mkv".format(video_name_cntr))

	os.rename(file_path, new_file_path)

	filename_mapping_dict[file_path] = video_name_cntr
	mapping_txt_file.write("{0}\t{1}\n".format(video_name_cntr, file_path))

	video_name_cntr+=1


mapping_txt_file.close()
pickle.dump(filename_mapping_dict, open(join(parent_dir, "filename_mapping_dict.p"), "wb"))


