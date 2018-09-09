editannotations.py

The arguments have been explained as follows :-

--modeYear : train/minval/val and year(depending on the MSCOCO dataset that you downloaded)
--hardNegDir : The directory containing all the mined hard negatives
--dataDir : Data directory of the py-faster-rcnn-ft repo

--destDir : Path where you would like to move the older annotations file. Note : This directory shouldn't exist at first. If you do another iteration of adding more hard negatives, then you may use the existing directory that you had used for the first time. 

If you want to use this feature, then uncomment line 27

--catId : MS COCO category id for the category


Usage example :-

python editannotations.py --modeYear train2014 --hardNegDir /mnt/nfs/scratch1/aprasad/train_detection_hard_negatives_attempt2/iter1_additional_hn_from_set3_videos/hardNegative_Jul16TrainSet3_Caffe+OHEM_alpha=5_beta=100_iouThr=0.20_validScore=0.85_validSimScore=0.50/train_detection_videos --dataDir /mnt/nfs/scratch1/aprasad/py-faster-rcnn-ft/data --destDir /mnt/nfs/scratch1/aprasad/coco_train_detection_iterations_attempt2 --catId 7


editannotations_source_dest.py

This version is to be used when you want the new annotations file at a different destination than the coco folder inside the data directory of the py-faster-rcnn-ft repo. 