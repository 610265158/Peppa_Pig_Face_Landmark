# Peppa_Pig_Face_Engine


## introduction

Purpose: I want to make a face analyzer including face detect and face alignment. Most of the face keypoints codes that opensourced are neither stable nor smooth, including some research papers. And the commercial sdk is pretty expensive. So, there is **Peppa_Pig_Face_Engine**.  



It is a simple demo including face detection and face aligment, and some optimizations were made to make the result smooth.


I think it is pretty cool, see the demo:

click the gif to see the video:
[![demo](https://github.com/610265158/simpleface-engine/blob/master/figure/sample.gif)](https://v.youku.com/v_show/id_XNDM3MTY4MTM2MA==.html?spm=a2h3j.8428770.3416059.1)


## useage

1. download pretrained model, put them into ./model
+ detector.pb

   +[baiduyun](https://pan.baidu.com/s/1DzbFYjcjcbXO4C494IB2TA) (code eb6b )
   
   +[googledrive](https://drive.google.com/drive/folders/1mV7I9UR_DjF91Wd2P6TqMQhMIOpcBWRJ?usp=sharing) 
+ keypoints.pb

    +[baiduyun](https://pan.baidu.com/s/1jPW9cq9V9sJDrcrtcqpmLQ)  (code wd5g)
    
    +[googledrive](https://drive.google.com/drive/folders/1YHtaLkalAqURbkIYYJBLf6HJZzd6vzOG?usp=sharing)
2. run `python demo.py --cam_id 0`    
   or  `python demo.py --video test.mp4`


##  Train
The project is based on two of my other repos. If you want to train with your own data, or you want to know the details about the models, click them.

 + [faceboxes](https://github.com/610265158/faceboxes-tensorflow.git)
 + [face_landmark](https://github.com/610265158/face_landmark.git)


## ps
The project is supported by myself, i need your advice to improve it.
Hope the codes can help you, contact me if u have any question( 2120140200@mail.nankai.edu.cn), and i need your star,also your contribution.

