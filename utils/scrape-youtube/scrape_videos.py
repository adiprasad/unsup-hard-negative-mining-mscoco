import youtube_dl
import pickle

#videos_list = pickle.load(open('chunk.p'))

videos_list = ['9PPDm86PlHM', 'u3M5AlqSG9c', '5MbUQJPwafI', '-SwDAkjQ9YU', '_WHeUKZTkTo', 'zWrKQzKqGdg', '4w1OeETx1ls', 'WOv7UsWnAuo', 'UuVQ9YCccHA', 'mgGdkJzcVMw', 'DJmeyxKwFCs', '5872c0YBy0k', 'Fx0vDGszhJg', 'ZjaHN5ehKew', 'NG1iTMJPs9Y', 'X3FmNVGZNCI', 'ActtcTqU-gs', 'UZ4RwQC3mPA', 'u_K_ywnu_rk', 'Uky5IH7BvcI', 'VLHe66EY2NM', '-BH6FnHdiCM', 'n7M5GDkiKck', 'E7hwzpY_5JI', 'HLElOTg5piM', 'SLR6nLZKoRU']

failed_videos_file = open('failed_videos.txt', 'a')

options = {'outtmpl': '%(id)s.%(ext)s', 'recodevideo' : 'mkv', 'merge_output_format' : 'mkv'}
youtube_prefix = "https://www.youtube.com/watch?v="

for key in videos_list:
	try:
		video_url = youtube_prefix + key
		with youtube_dl.YoutubeDL(options) as ydl:
			ydl.download([video_url])
	except Exception as e:
		print(e)
		print("Video {0} unavailable for download..Saved to file".format(key))
		failed_videos_file.write(key + "\n")


failed_videos_file.close()
