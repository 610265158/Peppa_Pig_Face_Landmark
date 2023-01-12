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

pretrained model is placed in pretrained, in Peppa_Pig_Face_Landmark folder.



## metric

WLFW test set.

| model   | Resolution | NME(test set) | model size (int8 weights) | Pretrained                                                   |
| ------- | ---------- | ------------- | ------------------------- | ------------------------------------------------------------ |
| Student | 128x128    | 5.03          | 2.0M                      |                                                              |
| Teacher | 128x128    | 4.65          | 7.5M                      |                                                              |
| Student | 256x256    | 4.65          | 2.0M                      | [model256](https://drive.google.com/drive/folders/1JFVrbMx07PwL47dFlUSZ1tAMcVxVmJXo?usp=share_link) |
| Teacher | 256x256    | 4.47          | 7.5M                      | [model256](https://drive.google.com/drive/folders/1JFVrbMx07PwL47dFlUSZ1tAMcVxVmJXo?usp=share_link) |



I will release new model when there is better one. 7.5K trainning data is not enough for a commercial used model. Please label more data if needed.

## requirment

+ pytorch

+ opencv

+ python 3.7

+ timm

  

## useage

### train

##### data

1. Download [WFLW](https://wywu.github.io/projects/LAB/WFLW.html) data. Set them in train_config.py.
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



