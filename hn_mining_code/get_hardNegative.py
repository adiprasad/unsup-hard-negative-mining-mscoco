"""
Code to mine hard negatives and identify and store frames containing hard negatives, with bounding box and confidence score info.

    Author: SouYoung Jin
"""


import cv2
import numpy as np
import os, glob
import scipy.io as sio 
import argparse
import pdb
import datetime

from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
from skimage import color

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Hard Negative Mining in Videos')
    parser.add_argument('--videoPath', dest='videoPath', help='Path of the video file')
    parser.add_argument('--detectionsFilePath', dest='detFilePath', help='Path of the classifier detections on the video')
    #parser.add_argument('--modelName', dest='modelName', help='Caffe, Caffe+OHEM, Caffe2', default='Caffe+OHEM')
    parser.add_argument('--validScore', dest='validScore', help='Valid detection confidence score', default='0.8')
    parser.add_argument('--alpha', dest='alpha', help='How many adjacent frames are going to be considered? ', default='5')
    parser.add_argument('--validSimScore', dest='validSimScore', help='validSimScore', default='0.5')
    parser.add_argument('--iouThr', dest='iouThr', help='iouThr', default='0.2')
    parser.add_argument('--beta', dest='beta', help='By "beta" pixels, dialate the current bounding box, and consider the box as a search region.', default='100')
    args = parser.parse_args()

    return args


def get_larger_box(curBox, W, H, beta):
    # let's make the region of search larger for the current box. 
    if beta == 0: return curBox
    left,top,right,bottom = curBox
    centerX = (left+right)/2
    centerY = (top+bottom)/2
    width = right-left+1
    height = bottom-top+1

    largerBox = np.zeros(4)
    largerBox[0], largerBox[2] = max(0,centerX-(width/2+beta)), min(W-1,centerX+(width/2+beta))
    largerBox[1], largerBox[3] = max(0,centerY-(height/2+beta)), min(H-1,centerY+(height/2+beta))
    return largerBox


def bb_intersection_over_union(boxA, boxB): 
    # @ "boxA" and "boxB" includes [left, top, right, bottom]

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # check if boxes are completed separated. 
    if xB-xA < 0 or yB-yA <0: 
        return 0   

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    denominator = float(boxAArea + boxBArea - interArea)
    iou = 0
    if denominator == 0: iou = 0
    else: iou = interArea / denominator
    
    # return the intersection over union value
    return iou


def get_date_today():
    month = datetime.date.today().strftime("%B")
    day = datetime.date.today().strftime("%d")
    year = datetime.date.today().strftime("%Y")

    return '-'.join([day, month, year])


'''============================================================================================================================================================================'''
def get_hardNegative(args):

    VIDEO_NAME = args.videoPath
    DATE_TODAY = get_date_today()
    print(VIDEO_NAME)
    print(DATE_TODAY)

    # *** Parameters for mining algorithm ***
    alpha = int(args.alpha) # How many adjacent frames are going to be considered? 
    beta = int(args.beta) # By "beta" pixels, dialate the current bounding box, and consider the box as a search region.
    iouThr = float(args.iouThr) # if IOU(curBox, adjBox) >= "iouThr", we will say two boxes are overlapped.  
    validScore = float(args.validScore) # Any face detection results of which final score is greater than equal to "validScore" will be interests of my algorithm.
    validSimScore = float(args.validSimScore) # a threshold to say two patches are quite different.
    crossBoundary = 50 # Any detections located on cross boundary will not consider as false positives
    smallBoxSize = 40*40 # Any detections whose size is smaller than "smallBoxSize" will not be considered.

    # ** Parameters for visualization ** 
    detectionColor = (0,255,255) # all detection boxes (yellow,4)
    negativeColor = (0,0,255) # current detection detected as hard negative (red,7)
    positiveColor = (0,255,0) # current detection NOT detected as hard negative (green,7)
    searchRegionColor = (255,255,0) # search region (cyan,4)
    trackletColor = (255,50,0) # best matching region, i.e. tracklet (light blue,4)
    fontSize = 0.6

    # *** Read a video ***
    FILE_PATH = VIDEO_NAME #os.path.join('./pedestrian_detection_videos_res/', VIDEO_NAME)
    VIDEO_ID = int(VIDEO_NAME[VIDEO_NAME.rfind('video')+5:])
    print('>> %s [ID:%d]' %(VIDEO_NAME, VIDEO_ID))
    

    # This is to read a ".mkv" video file, and to check the "alpha" adjacent frames
    # @@ IF YOU NEED TO READ OTHER VIDEO FILE, PLEASE WRITE DOWN YOUR OWN VERSION 
    for extension in ['*.mkv','*.MP4','*.mp4','*.avi']:    
        videoFileName = glob.glob(os.path.join(FILE_PATH,extension))
        if videoFileName != []: break
    
    print(videoFileName)
    vidcap = cv2.VideoCapture(videoFileName[0])

    # Check if camera opened successfully
    if (vidcap.isOpened()== False): 
        print("Error opening video stream or file")

    # @@

    # *** Some parameters for output results *** 
    VERSION = '%s_alpha=%d_beta=%d_iouThr=%.2f_validScore=%.2f_validSimScore=%.2f' % (DATE_TODAY,alpha,beta,iouThr,validScore,validSimScore)
    OUTPUT_PATH = './hardNegative_%s' %(VERSION)    # Frames containing hard negatives will be saved as ".jpg" in the "OUTPUT_PATH" 
                                                    # and a ".txt" file will be saved with the hard negative mining results.
    VISUALIZATION_PATH = './visualization_%s' %(VERSION)    # Hard negatives will be visualized.
   
    print("Output path")
    print(OUTPUT_PATH) 
    if not os.path.exists(os.path.join(OUTPUT_PATH,VIDEO_NAME)): os.makedirs(os.path.join(OUTPUT_PATH,VIDEO_NAME))
    if not os.path.exists(os.path.join(VISUALIZATION_PATH,VIDEO_NAME)): os.makedirs(os.path.join(VISUALIZATION_PATH,VIDEO_NAME))
    
    print(os.path.join(OUTPUT_PATH,VIDEO_NAME))
    '''---------------------------------------------------------------------------------------------------------------
    Get detection results  
    --------------------------------------------------------------------------------------------------------------- '''
    # All detection results in a video will be saved in "detBoxList"
    # E.g. 
    #   detBoxList[0] will show all detecion results on the first frame. 
    #   detBoxList[100] will show all detecion results on the 101-th frame. 
    
    # *** Read .txt file that includes detection results ***
    # @@ IF YOUR DETECTION HAS OTHER FORMAT, PLEASE WRITE DOWN YOUR OWN VERSION 

    '''
    if MODEL_NAME=='Caffe2':
        txtFile = open('./Caffe2VideoFaceDetection/%s_e2e_faster_rcnn_vgg16.txt' %(VIDEO_NAME))
    elif MODEL_NAME=='Caffe+OHEM':
        txtFileName = os.path.join('./dog_detection_outputs/', '%s.txt' %FILE_PATH)
        print(txtFileName)
        txtFile = open(txtFileName,'r')
    elif MODEL_NAME=='Caffe':
        txtFile = open(os.path.join(FILE_PATH,'results.txt'),'r')
    elif MODEL_NAME=='Retina':
        txtFileName = os.path.join('videos_similar_WIDER_R50','video%d.mp4_retinanet_R-50-FPN_1x.txt'%VIDEO_ID)
        if os.path.isfile(txtFileName):
            txtFile = open(txtFileName,'r')
        else:
            txtFile = open(os.path.join('frd_det_res','video%d_retinanet_R-50-FPN_1x.txt'%VIDEO_ID),'r')
    '''

    #txtFileName = os.path.join('./detection_outputs/', '%s.txt' %FILE_PATH)
    txtFileName = args.detFilePath
    print(txtFileName)
    txtFile = open(txtFileName,'r')


    # To parse "faster-rcnn" style output .txt files. 
    isNewFrame = True
    isReadingBox = False
    boxCount = 0
    frameCount = 0
    detBoxList = [] 
    boxList = []
    
    for line in txtFile:
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
            if score >= validScore:
                boxList.append([left,top,right,bottom,score])
            boxCount -= 1

        if isNewFrame == False and isReadingBox == True and boxCount == 0: 
            detBoxList.append(boxList)
            isReadingBox = False
            isNewFrame = True
            boxList = []
    # @@
                        
    
    '''---------------------------------------------------------------------------------------------------------------
    Detect isolated patches from videos. For +/- "alpha" consecutive frames, we will see if IOU between the adjacent frames is greater than "iouThr"             
    --------------------------------------------------------------------------------------------------------------- '''
    # *** Patterns of True Positives (Face) that are considered as "hard negatives" according to my previous algorithm. ***
    # 1) A person moves so quickly so that IOU(curBox, adjBox) becomes 0. 
    #                              ==> I increase the current box with "beta" pixels. 
    # 2) The face is occluded, so only the frame detects the face while the adjacent frames fail to do that. 
    #                              ==> Check if there's any occlusion or not, 
    #                                  by computing the similarity between the curBoxImage and adjacent frames cropped with the curBox.  
    # 3) Faces with extreme poses (upside down). These faces are occationally detected as faces by the face detection model. 
    

    # *** Detection results on frames containing hard negatives will be stored in "result" ***
    # @@ DO NOT CHANGE ANY CODE BELOW except for the purpose of reading and visualization 
    numFrame = len(detBoxList)
    if VIDEO_ID != 201:
        numFrame = min(numFrame,70000)
    result = []
    for count in range(0,numFrame):
        # print(count)
        if count == 0:
            _,image = vidcap.read(count) 
            H,W = image.shape[:2] # get the height and width of this video frame.

        # check if there are any detections having scores higher than "validScore"
        anyValidDetection = False
        boxCount = 0
        for box in detBoxList[count]:
            if box[4] >= validScore: 
                anyValidDetection = True
                break
            boxCount += 1

        # Skip this frame if there is 1) no valid detection results or 2) if the frame is at the beginning or end of the video
        if anyValidDetection == False or count-alpha+1 < 0 or count+alpha >= numFrame: 
            continue 
        
        # Read video frames (previous, current, next) in order to check similarity between the current box and the adjacent boxes.
        vidcap.set(1,count)
        _,image = vidcap.read()    
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # change "image" into a gray image
        
        bgnInd,endInd = max(0,count-alpha),min(numFrame-1,count+alpha)
        
        # Each box coordinate (x_min,y_min,width,height) on the current frame, i.e. "curBox", will be updated to the new coordinate, i.e. "adjBox", 
        # where IOU(curBox,adjBox) is greater than "iouThr" and adjBox is the argMax. 
        # If there are no "adjBox" on the previous "alpha" frames that have overlaps between "curBox", "outputPrev" for the detection will be remained False. 
        # For a detection on the current frame, if both "outputPrev" and "outputNext" are False, we will finally call the detection as a hard negative example.
        numCurBox = len(detBoxList[count]) 
        output = np.full(numCurBox,True,dtype=bool)
        outputPrev = np.full(numCurBox,False,dtype=bool)
        outputNext = np.full(numCurBox,False,dtype=bool) 
        prevImageList = []
        nextImageList = []
        for ci in range(0,numCurBox):
            curBox = detBoxList[count][ci][0:4]
            curScore = detBoxList[count][ci][4] # final classification score for the current coordinate   
            curWidth, curHeight = curBox[2]-curBox[0]+1, curBox[3]-curBox[1]+1
            
            # We will automatically consider the detection as faces 
            # 1) if <curScore> is very high, 2) the size of current detection is very small, or 3) cross boundary         
            if curScore >= 0.98 or curScore < validScore \
                or curWidth*curHeight < smallBoxSize \
                or curBox[0]<crossBoundary or curBox[2]>=W-crossBoundary or curBox[1]<crossBoundary or curBox[3]>=H-crossBoundary:
                outputPrev[ci],outputNext[ci] = True,True
                continue
            
            # Let's save <count-"alpha":count+"alpha"> frames as grayscale into "prevImageList" and "nextImageList"
            if len(prevImageList) == 0:
                for adjCount in range(count-1,bgnInd-1,-1):
                    vidcap.set(1,adjCount)
                    _,prevImage = vidcap.read()
                    prevImage = cv2.cvtColor(prevImage, cv2.COLOR_BGR2GRAY)
                    prevImageList.append(prevImage)
                for adjCount in range(count+1,endInd+1):
                    vidcap.set(1,adjCount)
                    _,nextImage = vidcap.read() 
                    nextImage = cv2.cvtColor(nextImage, cv2.COLOR_BGR2GRAY)
                    nextImageList.append(nextImage)   
                    
                
            # 1) For the region of search on the prev/next frames, find the best similar looking region to the curBox.  
            # In other words, search for the region on the adjacent frames that looks most similar to the curBox. 
            # Search space is "beta" pixels larger tha curBox to the left/right/bottom/top.
            # 2) Then, find if the region has overlapped with any of boxes on the prev/next frames.
            # 3) If the best similarity score is smaller than "validSimScore", we will skip considering the curBox as false positive.
            
            # Get the image of the current box
            curBox = detBoxList[count][ci][0:4]
            curBoxImage = grayImage[:]
            curBoxImage = curBoxImage[int(curBox[1]):int(curBox[3]), int(curBox[0]):int(curBox[2])]
            curBoxHeight, curBoxWidth = curBoxImage.shape[:2]

            # Get the search region with respect to the current box            
            searchRegion = map(int, get_larger_box(curBox,W,H,beta)) 
            searchRegionWidth = searchRegion[2] - searchRegion[0] + 1 - curBoxWidth
            searchRegionHeight = searchRegion[3] - searchRegion[1] + 1 - curBoxHeight

            # * Simple solution first
            #   -- frame <count-"alpha">:<count-1> 
            aci = -1
            for adjCount in range(count-1,bgnInd-1,-1):
                aci += 1
                numAdjBox = len(detBoxList[adjCount])
                if numAdjBox == 0: 
                    continue 
                iouScore = np.zeros(numAdjBox)

                # find the best similar regions on the adjacent frame.
                prevImage = prevImageList[aci].copy()
                searchRegionImage = prevImage[searchRegion[1]:searchRegion[3]+1, searchRegion[0]:searchRegion[2]+1]
                
                matchResult = cv2.matchTemplate(searchRegionImage, curBoxImage, cv2.TM_CCOEFF_NORMED)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(matchResult)
                if maxVal < validSimScore: 
                    outputPrev[ci] = True 
                    break

                topLeft = maxLoc # left, top
                bottomRight = (topLeft[0]+curBoxWidth, topLeft[1]+curBoxHeight) # right, bottom
                trackedBox = [searchRegion[0]+topLeft[0],searchRegion[1]+topLeft[1],searchRegion[0]+bottomRight[0],searchRegion[1]+bottomRight[1]]

                for ai in range(0,numAdjBox):
                    adjBox = detBoxList[adjCount][ai][0:4]
                    iouScore[ai] = max(bb_intersection_over_union(adjBox,trackedBox),bb_intersection_over_union(adjBox,curBox))
                if max(iouScore) > iouThr:
                    outputPrev[ci] = True
                    break
            
            if outputPrev[ci] == True: continue

            #   -- frame <count+1>:<count+"alpha">:
            aci = -1
            for adjCount in range(count+1,endInd+1):
                aci += 1
                numAdjBox = len(detBoxList[adjCount])
                if numAdjBox == 0: 
                    continue 
                iouScore = np.zeros(numAdjBox)

                # find the best similar regions on the adjacent frame.
                nextImage = nextImageList[aci].copy()
                searchRegionImage = nextImage[searchRegion[1]:searchRegion[3]+1, searchRegion[0]:searchRegion[2]+1]
                
                matchResult = cv2.matchTemplate(searchRegionImage, curBoxImage, cv2.TM_CCOEFF_NORMED)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(matchResult)
                if maxVal < validSimScore: 
                    outputNext[ci] = True 
                    break

                topLeft = maxLoc
                bottomRight = (topLeft[0]+curBoxWidth, topLeft[1]+curBoxHeight)
                trackedBox = [searchRegion[0]+topLeft[0],searchRegion[1]+topLeft[1],searchRegion[0]+bottomRight[0],searchRegion[1]+bottomRight[1]]
                
                for ai in range(0,numAdjBox):
                    adjBox = detBoxList[adjCount][ai][0:4]
                    iouScore[ai] = max(bb_intersection_over_union(adjBox,trackedBox),bb_intersection_over_union(adjBox,curBox))
                if max(iouScore) > iouThr:
                    outputNext[ci] = True
                    break
            
            if outputPrev[ci] == False and outputNext[ci] == False:
                output[ci] = False
        

        # Visualize the results for the frames containing false positives
        # # ** Parameters for visualization ** 
        # detectionColor = (0,255,255) # all detection boxes (yellow,4)
        # negativeColor = (0,0,255) # current detection detected as hard negative (red,7)
        # positiveColor = (0,255,0) # current detection NOT detected as hard negative (green,7)
        # searchRegionColor = (255,255,0) # search region (cyan,4)
        # trackletColor = (255,50,0) # best matching region, i.e. tracklet (light blue,4)
        # fontSize = 0.6
        if not all(output):
            boxColor = ''
            fontColor = (0,0,0)
            curFileName = os.path.join(VIDEO_NAME,'frame%d.jpg'%(count+1))
            curBoundingBox = []
            curIsFace = []

            # save the original image
            cv2.imwrite(os.path.join(OUTPUT_PATH,curFileName), image)

            # save the labeled image
            boxCount = 0 
            for box in detBoxList[count]:
                if box[4] >= validScore:
                    curBoundingBox.append([box[0],box[1],box[2]-box[0],box[3]-box[1],box[4]]) # x_min(left),y_min(top),width,score
                    if output[boxCount] == True:                 
                        curIsFace.append(1)
                        boxColor = positiveColor
                    else: 
                        curIsFace.append(0)
                        boxColor = negativeColor
                    
                    cv2.rectangle(image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),boxColor,7)
                    cv2.rectangle(image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),boxColor,7)
                    cv2.putText(image,str(format(box[4],'.2f')),(int(box[0]),int(box[3]+20)),cv2.FONT_HERSHEY_SIMPLEX,fontSize,boxColor,1,cv2.LINE_AA)
                
                boxCount += 1

            cv2.imwrite(os.path.join(VISUALIZATION_PATH,curFileName), image)
            
            # add the detection results from the current frame
            curResult = {'fileName':curFileName, 'boundingBox':curBoundingBox, 'isFace':curIsFace}
            result.append(curResult)
            print('%.4f' %(float(count+1)/numFrame))


    # *** Save the result ***
    outputTxtFile = open(os.path.join(OUTPUT_PATH,'%s.txt'%(VIDEO_NAME)), 'w')

    for curResult in result: 
        outputTxtFile.write(curResult['fileName']+'\n')
        numBox = len(curResult['boundingBox'])
        outputTxtFile.write(str(numBox)+'\n')
        for b in xrange(numBox):
            curBox = curResult['boundingBox'][b]
            outputTxtFile.write('%f %f %f %f %d\n' %(curBox[0],curBox[1],curBox[2],curBox[3],curResult['isFace'][b]))
    outputTxtFile.close()

    print('  --- number of frames containing false positives: %d' % len(result)) 

'''============================================================================================================'''
if __name__ == '__main__':
    args = parse_args()
    get_hardNegative(args)

