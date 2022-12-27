# face_landmark
A simple face aligment method, based on pytorch


## introduction


It is simple and flexible, trained with wingloss , multi task learning, also with data augmentation based on headpose and face attributes(eyes state and mouth state).

[CN blog](https://blog.csdn.net/qq_35606924/article/details/99711208)

The model is trained for **[[pappa_pig_face_engine]](https://github.com/610265158/Peppa_Pig_Face_Engine).**

Contact me if u have problem about it. 2120140200@mail.nankai.edu.cn :)

demo pictures:

![samples](https://github.com/610265158/face_landmark/blob/master/figures/tmp_screenshot_18.08.20192.png)

![gifs](https://github.com/610265158/Peppa_Pig_Face_Engine/blob/master/figure/sample.gif)

this gif is from github.com/610265158/Peppa_Pig_Face_Engine )

pretrained model is placed in pretrained, in Peppa_Pig_Face_Engine folder.



## requirment

+ pytorch

+ opencv

+ python 3.7

+ timm

  

## useage

### train

##### data

1. The data are from 300W and 300VW, and i crop them to samll size for fast read. You could download them from [google drive](https://drive.google.com/drive/folders/1R26vvQNQh9E5MXP50bAo4gPm1OdjDrko?usp=share_link). Or you can prepare them by yourself.
3. then  `run.sh`

4. by default it is trained with mobilenetv3-large as backbone.

### visualization

```
python vis.py --model ./keypoints.pth
```



### convert to onyx

``` python
python tools/convert_to_onnx.py --model ./keypoints.pth
```



