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


| WFLW    | inputsize | NME     | Flops(G)| Params(M)  | Pose | Exp. | Ill. | Mu.  | Occ. | Blur | pretrained                                                                                   |
|---------|-----------|---------|----------|---------|------|------|------|------|------|------|----------------------------------------------------------------------------------------------|
| Student | 128x128   |         |          |         |      |           |      |      |      |      |                                                                                              |
| Teacher | 128x128   |         |          |         |      |         |      |      |      |      |                                                                                              |
| Student | 256x256   | **4.35**    | 1.39     | 3.25    | 7.53 | 4.52    | 4.16 | 4.21 | 5.34 | 4.93 | [skps](https://drive.google.com/drive/folders/1Y8FvJV1X5YTUkwt5MywVFvqzStpxRK_S?usp=sharing) |
| Teacher | 256x256   | **3.95**    | 5.53     | 11.53   | 7.00 | 4.00    | 3.81 | 3.78 | 4.85 | 4.54 | [skps](https://drive.google.com/drive/folders/1Y8FvJV1X5YTUkwt5MywVFvqzStpxRK_S?usp=sharing) |



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



