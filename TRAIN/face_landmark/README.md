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

Update, new model!!

| model   | Resolution | NME(test set) | Params | Flops | Pretrained                                                                                            |
| ------- |------------|---------------|--------|-------|-------------------------------------------------------------------------------------------------------|
| Student | 128x128    |           | 2.07M  | 0.63G |    |
| Teacher | 128x128    |           | 27.42M | 1.30G |        |
| Student | 256x256    | 4.60          | 2.07M  | 2.49G | [model256_update](https://drive.google.com/file/d/1bLE6RhQu-SVmQR2MWIt6s2vyUbnTjA83/view?usp=sharing)        |
| Teacher | 256x256    | 4.24          | 27.42M | 5.18G | [model256_update](https://drive.google.com/file/d/1bLE6RhQu-SVmQR2MWIt6s2vyUbnTjA83/view?usp=sharing) |


| WLFW    | inputsize | Fullset | Pose | Exp. | Ill. | Mu.  | Occ. | Blur |
|---------|-----------|--------|------|------|------|------|------|------|
| Student | 128x128   |     |  |  |  |  |  |  |
| Teacher | 128x128    |     | |  |  |  |  |  |
| Student | 256x256   | 4.60   | 7.84 | 4.71 | 4.40 | 4.49 | 5.90 | 5.31 |
| Teacher | 256x256   | 4.24   | 7.06 | 4.27 | 4.10 | 4.03 | 5.28 | 4.90 |


## requirment

+ pytorch

+ opencv

+ timm

  

## useage

### train

##### data

1. Download [WFLW](https://wywu.github.io/projects/LAB/WFLW.html) data. Set them in train_config.py.
3. then  `run.sh`

4. by default it is trained with mobilenetv3-large as student, efficientnetb5 as teacher.

### visualization

```
python vis.py --model ./keypoints.pth
```



### convert to onyx

``` python
python tools/convert_to_onnx.py --model ./keypoints.pth
```



