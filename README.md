# Peppa_Pig_Face_Engine
[![DOI](https://zenodo.org/badge/206305226.svg)](https://zenodo.org/badge/latestdoi/206305226)



## introduction

It is a simple demo including face detection and face aligment, and some optimizations were made to make the result better.


The keypoint model encodes and decodes the x and y coordinates using heatmap and offset of x and y, 
achieving SOTA on WFLW dataset. 
Like object detection, heatmap predicts which point is a positive sample on the featuremap, 
represented as a highlighted area, while x and y offsets are responsible for predicting the specific coordinates of these positive samples.
And it achieves NME 3.95 on WFLW with no extern data.

click the gif to see the video:
[![demo](https://github.com/610265158/simpleface-engine/blob/master/figure/sample.gif)](https://v.youku.com/v_show/id_XNDM3MTY4MTM2MA==.html?spm=a2h3j.8428770.3416059.1)

and with face mask:
![face mask](https://github.com/610265158/Peppa_Pig_Face_Engine/blob/master/figure/sample_mask.gif)

## requirment

+ PyTorch
+ onnxruntime  
+ opencv
+ easydict

## model

### 1 face detector

  [yolov5-face](https://github.com/deepcam-cn/yolov5-face)

### 2 landmark detector
    
###### HOW TO TRAIN
  [simple face landmark detector]( https://github.com/610265158/Peppa_Pig_Face_Landmark/tree/master/TRAIN/face_landmark) 

  Refer to [TRAIN/face_landmark/README.md](https://github.com/610265158/Peppa_Pig_Face_Landmark/blob/master/TRAIN/face_landmark/README.md) to train the model.

| WFLW    | inputsize | NME      | Flops(G) | Params(M) | Pose | Exp. | Ill. | Mu.  | Occ. | Blur | pretrained                                                                                      |
|---------|-----------|----------|----------|-----------|------|------|------|------|------|------|-------------------------------------------------------------------------------------------------|
| Student | 128x128   | **4.80** | 0.35     | 3.25      | 8.53 | 5.00 | 4.61 | 4.81 | 5.80 | 5.36 | [skps](https://drive.google.com/drive/folders/1JktGIKohpeLO14a6eJqNlZort_46qVC0?usp=share_link) |
| Teacher | 128x128   | **4.17** | 1.38     | 11.53     | 7.14 | 4.32 | 4.01 | 4.03 | 4.98 | 4.68 | [skps](https://drive.google.com/drive/folders/1JktGIKohpeLO14a6eJqNlZort_46qVC0?usp=share_link) |
| Student | 256x256   | **4.35** | 1.39     | 3.25      | 7.53 | 4.52 | 4.16 | 4.21 | 5.34 | 4.93 | [skps](https://drive.google.com/drive/folders/1Y8FvJV1X5YTUkwt5MywVFvqzStpxRK_S?usp=sharing)    |
| Teacher | 256x256   | **3.95** | 5.53     | 11.53     | 7.00 | 4.00 | 3.81 | 3.78 | 4.85 | 4.54 | [skps](https://drive.google.com/drive/folders/1Y8FvJV1X5YTUkwt5MywVFvqzStpxRK_S?usp=sharing)    |


  I will release new model when there is better one. 7.5K trainning data is not enough for a very good model. Please label more data if needed.

## useage

1. pretrained models are in ./pretrained, for easy to use ,we convert them to mnn
2. run `python demo.py --cam_id 0` use a camera    
   or  `python demo.py --video test.mp4`  detect for a video    
   or  `python demo.py --img_dir ./test`  detect for images dir no track   
   or  `python demo.py --video test.mp4 --mask True` if u want a face mask



```python
# by code:
from lib import FaceAna
facer = FaceAna()
boxes, landmarks, _ = facer.run(image)
  
```



