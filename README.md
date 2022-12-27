# Peppa_Pig_Face_Engine



## introduction

It is a simple demo including face detection and face aligment, and some optimizations were made to make the result smooth.



Face detector is from [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)

Face landmark training codes are placed in folder  TRAIN/face_landmark.




I think it is pretty cool, see the demo:

click the gif to see the video:
[![demo](https://github.com/610265158/simpleface-engine/blob/master/figure/sample.gif)](https://v.youku.com/v_show/id_XNDM3MTY4MTM2MA==.html?spm=a2h3j.8428770.3416059.1)

and with face mask:
![face mask](https://github.com/610265158/Peppa_Pig_Face_Engine/blob/master/figure/sample_mask.gif)

## requirment

+ PyTorch （tensorflow1 need to switch to tf1 branch )

+ MNN  

+ opencv

+ python 3.7

+ easydict

  

## useage

1. pretrained models are in ./pretrained, for easy to use ,we convert them to men
2. run `python demo.py --cam_id 0` use a camera    
   or  `python demo.py --video test.mp4`  detect for a video    
   or  `python demo.py --img_dir ./test`  detect for images dir no track   
   or  `python demo.py --video test.mp4 --mask True` if u want a face mask



```python
# by code:
from lib.core.api.facer import FaceAna
facer = FaceAna()
boxes, landmarks, states = facer.run(image)
  
```



##  Train

The face landmarks training codes are placed in sub folder ./TRAIN/face_landmark

If you want to train the model ,please work in the sub folder ,refer to ./TRAIN/face_landmark/README.md


