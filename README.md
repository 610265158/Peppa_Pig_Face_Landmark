# Peppa_Pig_Face_Engine



## introduction

It is a simple demo including face detection and face aligment, and some optimizations were made to make the result better.




I think it is pretty cool, see the demo:

click the gif to see the video:
[![demo](https://github.com/610265158/simpleface-engine/blob/master/figure/sample.gif)](https://v.youku.com/v_show/id_XNDM3MTY4MTM2MA==.html?spm=a2h3j.8428770.3416059.1)

and with face mask:
![face mask](https://github.com/610265158/Peppa_Pig_Face_Engine/blob/master/figure/sample_mask.gif)

## requirment

+ PyTorch
+ MNN  
+ opencv
+ python 3.7
+ easydict

## model

+ 1 face detector

  [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)

+ 2 landmark detector

  [simple face landmark detector]( https://github.com/610265158/Peppa_Pig_Face_Landmark/tree/master/TRAIN/face_landmark) 

  Refer to [TRAIN/face_landmark/README.md](https://github.com/610265158/Peppa_Pig_Face_Landmark/blob/master/TRAIN/face_landmark/README.md) to train the model.

  the model is trained with WFLW data. For student **mobilenetv3-large** was used  as backbone, for teacher is **efficientnetb5**.

  | model   | Resolution | NME(test set) | model size (int8 weights) | Pretrained                                                   |
  | ------- | ---------- | ------------- | ------------------------- | ------------------------------------------------------------ |
  | Student | 128x128    | 4.95          | 1.9M                      | [model128](https://drive.google.com/drive/folders/1zivD151CkOSm8KYyeC7v4YPC0aYDomry?usp=share_link) |
  | Teacher | 128x128    | 4.64          | 6.9M                      | [model128](https://drive.google.com/drive/folders/1zivD151CkOSm8KYyeC7v4YPC0aYDomry?usp=share_link) |
  | Student | 256x256    | 4.65          | 1.9M                      | [model256](https://drive.google.com/drive/folders/1JFVrbMx07PwL47dFlUSZ1tAMcVxVmJXo?usp=share_link) |
  | Teacher | 256x256    | 4.47          | 6.9M                      | [model256](https://drive.google.com/drive/folders/1JFVrbMx07PwL47dFlUSZ1tAMcVxVmJXo?usp=share_link) |

  

  I will release new model when there is better one. 7.5K trainning data is not enough for a very good model. Please label more data if needed.

## useage

1. pretrained models are in ./pretrained, for easy to use ,we convert them to mnn
2. run `python demo.py --cam_id 0` use a camera    
   or  `python demo.py --video test.mp4`  detect for a video    
   or  `python demo.py --img_dir ./test`  detect for images dir no track   
   or  `python demo.py --video test.mp4 --mask True` if u want a face mask



```python
# by code:
from lib.core.api.facer import FaceAna
facer = FaceAna()
boxes, landmarks, _ = facer.run(image)
  
```



