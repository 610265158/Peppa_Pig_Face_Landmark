# Peppa_Pig_Face_Engine


## introduction

It is a simple demo including face detection and face aligment, and some optimizations were made to make the result smooth.

**CAUTION: this is the tensorflow2.0 branch, if you want to work on tf1, please switch to tf1 branch, it still works.**

Purpose: I want to make a face analyzer including face detect and face alignment. Most of the face keypoints codes that opensourced are neither stable nor smooth, including some research papers. And the commercial sdk is pretty expensive. So, there is **Peppa_Pig_Face_Engine**.  


I think it is pretty cool, see the demo:

click the gif to see the video:
[![demo](https://github.com/610265158/simpleface-engine/blob/master/figure/sample.gif)](https://v.youku.com/v_show/id_XNDM3MTY4MTM2MA==.html?spm=a2h3j.8428770.3416059.1)

and with face mask:
![face mask](https://github.com/610265158/Peppa_Pig_Face_Engine/blob/master/figure/sample_mask.gif)

## requirment

+ tensorflow2.0 （tensorflow1 need to switch to tf1 branch )

+ opencv

+ python 3.6

+ easydict

+ flask


## update

2020.2.4:  add http server

## useage

1. download pretrained model, put them into ./model
+ detector  

       Lightnet_0.5  including a tflite model, 
        (time cost: mac i5-8279U@2.4GHz， tf2.0 15ms+， tflite 8ms+-,input shape 320x320, model size 560K)
    + [baidu disk](https://pan.baidu.com/s/1ZJZHJz8VFXahmwBptGQfiA) ( password yqst )
    + [google drive](https://drive.google.com/open?id=1ZZVA7QhwGWYJ-09KoU2iym90zqbrfTQH)
   
+ keypoints

      shufflenetv2_0.75   including a tflite model, 
       (time cost: mac i5-8279U@2.4GHz， tf2.0 5ms+， tflite 3.7ms+- model size 2.5M)
    + [baidu disk](https://pan.baidu.com/s/1JxZ9nhFpWCAv5A44yUEcOA)  (code fcdc)
    + [google drive](https://drive.google.com/open?id=1VAJ8qObyRfLmpimoZA6QwrhXjQmgwBXn)


    the dir structure as :
    ```
       ./model
       ├── detector
       │   ├── saved_model.pb
       │   └── variables
       │       ├── variables.data-00000-of-00002
       │       ├── variables.data-00001-of-00002
       │       └── variables.index
       ├── keypoints
       │   ├── saved_model.pb
       │   └── variables
       │       ├── variables.data-00000-of-00002
       │       ├── variables.data-00001-of-00002
       │       └── variables.index
    ```
2. run `python demo.py --cam_id 0` use a camera    
   or  `python demo.py --video test.mp4`  detect for a video    
   or  `python demo.py --img_dir ./test`  detect for images dir no track   
   or  `python demo.py --video test.mp4 --mask True` if u want a face mask
    
## start a http server

1. run `python demo.py --web 1`

2. test by run `python web_demo_test.py`, 

the result is a json, formate:
`
[{ "bbox": [x1, y1, x2, y2], 
    "landmark": [[x1, y1], [x2, y2],[x3,y3],[...]]}]
`


##  Train
The project is based on two of my other repos, and both tensorflow1 and tensorflow2 are supported. 
If you want to train with your own data, 
or you want to know the details about the models, click them.

 + [dsfd_light_model](https://github.com/610265158/DSFD-tensorflow)  **tf2 branch**
 + [face_landmark](https://github.com/610265158/face_landmark.git)


## ps
The project is supported by myself,
and i need your advice to improve it.
Hope the codes can help you, 
contact me if u have any question.
At last, i need your star,also your contribution.

## TODO

- [x]  Transfer to tensorflow 2.0   
- [x]  small model including tflite
- [x]  add http server demo
- [ ]  Add some GAN model to make it fun ing....
- [ ]  3-d face algorithm
- [ ]  a mobile device version, it is on the way, i learn a lot about mobile device. **
- [ ]  switch to pytorch,  HAHA...   dont worry it wont happen.
