"""
Code to visualize detections on the supplied hard negative frames for a video
    Author: Deep Chakraborty
    Date Created: 04/25/2018
"""

import cv2
import argparse
import os
import re

def parse_args():
    """Returns dictionary containing CLI arguments"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--hn-frames", required=True, help="path to the folder containing hard negative raw frames")
    ap.add_argument("-d", "--detections", required=True, help="file containing faster R-CNN detections for the corresponding video")
    ap.add_argument("-o", "--out-path", required=True, help="path to the folder containing visualization of detections on HN frames")
    args = vars(ap.parse_args())
    return args


def parse_detections_file(detections_file_name):
    """Parses the detections file and gets a list of detections on each frame"""
    try:
        detections_file = open(detections_file_name, 'r')
    except IOError:
        raise("Detection File provided doesn't exist")

    # Adapted from SouYoung's hard negative mining code.
    isNewFrame = True
    isReadingBox = False
    boxCount = 0
    frameCount = 0
    detBoxList = [] 
    boxList = []
    
    for line in detections_file:
        # print line
        if isNewFrame == True and isReadingBox == False:
            isNewFrame = False
        elif isNewFrame == False and isReadingBox == False:
            boxCount = int(line)
            frameCount = frameCount + 1
            isReadingBox = True
        elif isNewFrame == False and isReadingBox == True:
            line = line.split()
            left,top,width,height,score = [float(l) for l in line]
            right,bottom = left+width,top+height
            left,top = left,top
            boxList.append([left,top,right,bottom,score])
            boxCount -= 1

        if isNewFrame == False and isReadingBox == True and boxCount == 0: 
            detBoxList.append(boxList)
            isReadingBox = False
            isNewFrame = True
            boxList = []

    return detBoxList


def _draw_detections(frame, frame_detections):
    """Draws rectangles on the frame using supplied detections"""
    boxColor = (0,255,0)
    for box in frame_detections:
        cv2.rectangle(frame,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),boxColor,7)
        # cv2.rectangle(frame,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),boxColor,7)
        cv2.putText(frame,str(format(box[4],'.2f')),(int(box[0]),int(box[3]+20)),cv2.FONT_HERSHEY_SIMPLEX,0.6,boxColor,1,cv2.LINE_AA)

    return frame


def vis_detections(frame_path, save_path):
    """Takes a single frame and draws detections on it"""
    global detections
    # Find the frame number.
    find_frame_num = re.compile(r'\d+')
    frame_num = int(find_frame_num.search(f).group(0))
    frame_detections = detections[frame_num]

    frame = cv2.imread(frame_path)
    frame_with_detections = _draw_detections(frame, frame_detections)
    cv2.imwrite(save_path, frame_with_detections)


if __name__ == '__main__':
    args = parse_args()
    detections_file_path = args["detections"]
    hn_frames_path = args["hn_frames"]
    outpath = args["out_path"]

    detections = parse_detections_file(detections_file_path)

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    for dirpath, dnames, fnames in os.walk(hn_frames_path):
        for f in fnames:
            print "Processing {} ...".format(f)
            if f.endswith(".jpg"):
                frame_path = os.path.join(dirpath, f)
                save_path = os.path.join(outpath, f)
                vis_detections(frame_path, save_path)