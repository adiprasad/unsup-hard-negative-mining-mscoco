# unsup-hard-negative-mining-mscoco
This is the repository for experiments on the MSCOCO classes mentioned in the paper [Unsupervised Hard Example Mining from Videos for Improved Object Detection](https://arxiv.org/abs/1808.04285) mentioned in Section 5(Discussion).

We used the original version of [py-faster-rcnn-ft](https://github.com/DFKI-Interactive-Machine-Learning/py-faster-rcnn-ft) to fine-tune the VGG16 network pretrained on ImageNet dataset to convert it to a binary classifier for an MSCOCO category. Once we had the classifier as the backbone network of the Faster RCNN, we used it to label all the frames within a video for the presence of that particular MSCOCO category. Using the labelled frames, we were able to identify the frames containing hard negatives with the help of our algorithm. Finally, we fine tuned the network again after including the frames containing hard negatives and evaluated the network for improvements using held out validation and test sets.

For our research, we carried out experiments on two MSCOCO categories, Dog and Train. 

## Steps :-

### 1. Preparing a Faster RCNN object detector on an MSCOCO category

Follow the steps mentioned in the [py-faster-rcnn-ft](https://github.com/DFKI-Interactive-Machine-Learning/py-faster-rcnn-ft) repository to prepare a VGG16 Faster RCNN network trained on an MSCOCO category of your choice.  

### 2. Label the videos with detections

Scrape the web and download videos that are likely to contain a lot of instances of your chosen category. Helper code to download youtube videos can be found [here](utils/scrape-youtube/scrape_videos.py). Once the videos have been downloaded, run the detections code to label each frame of every video with bounding boxes and confidence scores for that category. See [Usage](detections_code/README.txt) 

The list of videos we used is mentioned below :-

1. [Dog videos](https://docs.google.com/spreadsheets/d/1q9EeOHVYXugtmR1batdDDsb5wzWnwiQc-egLDmdWk78/#gid=1264294087) 
2. [Train videos](https://docs.google.com/spreadsheets/d/1q9EeOHVYXugtmR1batdDDsb5wzWnwiQc-egLDmdWk78/#gid=994319682)

### 3. Hard negative mining 

The detections code outputs a txt file containing frame wise labeling and bounding box information. Use the hard negative mining code on the detections txt file to output the frames containing hard negatives and a txt file containing the bounding box information on those frames. See [Usage](hn_mining_code/README.txt). 

### 4. Include the video frames containing hard negatives in the COCO dataset and fine-tune

Use the COCO annotations editor located inside utils to include the frames containing hard negatives in MSCOCO dataset. One the frames have been included in the COCO dataset, fine-tune to get an improved network. See [Usage](utils/edit-coco-annotations/README.txt)


## Results :-


<br>

A summary of the results is mentioned below :-

<br>
<table>
    <thead>
        <tr>
          <th><b>Category</b></th>
            <th><b>Model</b></th>
            <th><b>Training Iterations</b></th>
            <th><b>Training Hyperparams</b></th>          
            <th><b>Validation set AP</b></th>          
          <th><b>Test set AP</b></th>          
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>Dog</td>
            <td rowspan=1>Baseline</td>
            <td rowspan=1>29000</td>
            <td>LR : 1e-3 for 10k,<br>1e-4 for 10k-20k,<br>1e-5 for 20k-29k</td>
            <td rowspan=1>26.9</td>
            <td rowspan=1>25.3</td>
        </tr>
        <tr>
            <td rowspan=1>Flickers as HN</td>
            <td rowspan=1>22000</td>
            <td>LR : 1e-4 for 15k,<br>1e-5 for 15k-22k</td>
            <td rowspan=1>28.1</td>
            <td rowspan=1>26.4</td>
        </tr>
        <tr>
            <td rowspan=2>Train</td>
            <td rowspan=1>Baseline</td>
            <td rowspan=1>26000</td>
            <td>LR : 1e-3,<br>stepsize : 10k,<br>lr decay : 0.1</td>
            <td rowspan=1>33.9</td>
            <td rowspan=1>33.2</td>
        </tr>
        <tr>
            <td rowspan=1>Flickers as HN</td>
            <td rowspan=1>24000</td>
            <td>LR : 1e-3,<br>stepsize : 10k,<br>lr decay : 0.1</td>
            <td rowspan=1>35.4</td>
            <td rowspan=1>33.7</td>
        </tr>
    </tbody>
</table>

<br>
A few examples on the reduction in false positives achieved for the 'Dog' category are mentioned below :-
<br>
<br>

Baseline             |  Flickers as HN
:-------------------:|:--------------------:
![](https://people.cs.umass.edu/~aprasad/Detector_Results/dog_detector/images_iter_1/video1/hns_shown/frame330_before.jpg)  |  ![](https://people.cs.umass.edu/~aprasad/Detector_Results/dog_detector/images_iter_1/video1/hns_shown/frame330_after.jpg)
![](https://people.cs.umass.edu/~aprasad/Detector_Results/dog_detector/images_iter_1/video1/hns_shown/frame1548_before.jpg)  |  ![](https://people.cs.umass.edu/~aprasad/Detector_Results/dog_detector/images_iter_1/video1/hns_shown/frame1548_after.jpg)
![](https://people.cs.umass.edu/~aprasad/Detector_Results/dog_detector/images_iter_1/video1/hns_shown/frame3156_before.jpg)  |  ![](https://people.cs.umass.edu/~aprasad/Detector_Results/dog_detector/images_iter_1/video1/hns_shown/frame3156_after.jpg)
![](https://people.cs.umass.edu/~aprasad/Detector_Results/dog_detector/images_iter_1/video1/hns_shown/frame9195_before.jpg)  |  ![](https://people.cs.umass.edu/~aprasad/Detector_Results/dog_detector/images_iter_1/video1/hns_shown/frame9195_after.jpg)
![](https://people.cs.umass.edu/~aprasad/Detector_Results/dog_detector/images_iter_1/video1/hns_shown/frame43837_before.jpg)  |  ![](https://people.cs.umass.edu/~aprasad/Detector_Results/dog_detector/images_iter_1/video1/hns_shown/frame43837_after.jpg)
