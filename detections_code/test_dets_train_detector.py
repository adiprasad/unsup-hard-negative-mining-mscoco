import time
#from __future__ import division
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import sys

from glob import glob
from os import getcwd, chdir
from os.path import join

NETS = {'vgg16': ('VGG16',
          '/mnt/nfs/scratch1/aprasad/py-faster-rcnn-ft/output/faster_rcnn_end2end/coco_2014_train/vgg16_faster_rcnn_iter_32000.caffemodel')}

def list_files(directory, extensions):
    saved = getcwd()
    chdir(directory)
    its = []
    for extension in extensions:
        it = glob('*.' + extension)
        if it:
            its.append(it[0])
    chdir(saved)
    return its[0]


def vis_detections_video(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""

    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return im

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),2)
        cv2.rectangle(im,(int(bbox[0]),int(bbox[1])-10),(int(bbox[0]+200),int(bbox[1])+10),(10,10,10),-1)
        cv2.putText(im,'{:s} {:.3f}'.format(class_name, score),(int(bbox[0]),int(bbox[1]-2)),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255))#,cv2.CV_AA)
    return im

def demo_video(net, im,frame_number,args, output_folder, conf_thresh): 
    """Detect object classes in an image using pre-computed object proposals."""
     
    #data_dir = '/mnt/nfs/scratch1/souyoungjin/RESULTS_FACE_DETECTIONS/ARUNI/video'+str(args.video_id)
    #out_dir = '/mnt/nfs/scratch1/ashishsingh/RESULTS_DOG_DETECTIONS/ARUNI/OHEM/'+args.video_folder_name
    
    out_dir = join(output_folder, args.video_id)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    #CONF_THRESH = 0.80 #check threshold values
    CONF_THRESH = conf_thresh
    NMS_THRESH = 0.15  #check threshold values
    
    # detection file
    dets_file_name = os.path.join(out_dir, 'video'+str(args.video_id)+'.txt') 
    fid = open(dets_file_name, 'a+')
    sys.stdout.write('%s ' % (frame_number))
    
    cls_ind = 1
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]

    keep = np.where(dets[:, 4] > CONF_THRESH)
    dets = dets[keep]
    
    dets[:, 2] = dets[:, 2] - dets[:, 0] + 1
    dets[:, 3] = dets[:, 3] - dets[:, 1] + 1
    
    fid.write('FRAME NUMBER: '+ str(frame_number) + '\n')
    fid.write(str(dets.shape[0]) + '\n')
    for j in xrange(dets.shape[0]):
          fid.write('%f %f %f %f %f\n' % (dets[j, 0], dets[j, 1], dets[j, 2], dets[j, 3], dets[j, 4]))

    print ''
    fid.close()
    
    #for cls_ind, cls in enumerate(CLASSES[1:]):
        #cls_ind += 1 # because we skipped background
        #cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]

        #cls_scores = scores[:, cls_ind]
        #dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
        #keep = nms(dets, NMS_THRESH)
        #dets = dets[keep, :]
        #im=vis_detections_video(im, cls, dets, thresh=CONF_THRESH)
    #cv2.imwrite(os.path.join('output',str(time.time())+'.jpg'),im)
    
    
    #cv2.imshow('ret',im)
    #cv2.waitKey(20)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--conf_thresh', dest='conf_thresh', help='Classification threshold',
                        default=0.85, type=float)
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--video_folder', dest='video_folder_name', help='video folder name',
                        type=str)
    parser.add_argument('--video', dest='video_id', help='video number',
                        type=str)
    parser.add_argument('--out_folder', dest='output_folder', help='output_folder',
                        type=str)


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    # cfg.TEST.BBOX_REG = False

    args = parse_args()

    #prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    #caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',NETS[args.demo_net][1])

    prototxt = 'models/coco/VGG16/faster_rcnn_end2end/test.prototxt'
    caffemodel = NETS[args.demo_net][1]

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
             'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
        net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    print("Args video id : {0}".format(args.video_id))
   
    
    ### Load Video File
  
    #videoFilePath = '/mnt/nfs/scratch1/souyoungjin/'+args.video_folder_name+'/video'+str(args.video_id)+'/101.mkv'
    videoFilePath = args.video_folder_name+'/video'+str(args.video_id)
    video_name = list_files(videoFilePath,['mkv','mp4'])
    videoFilePath = args.video_folder_name+'/video'+str(args.video_id)+'/'+video_name
    output_folder = args.output_folder
    conf_thresh = args.conf_thresh
    
    videoFile = cv2.VideoCapture(videoFilePath)
    count = 0
    while True:
        ret, image = videoFile.read()
	#print image.shape
        if not ret:
	    print "Video failed to read"
            break
        count +=1
        demo_video(net, image, count, args, output_folder, conf_thresh)
