# face_landmark

A simple face aligment method, based on pytorch


## introduction


The keypoint model encodes and decodes the x and y coordinates using heatmap and offset of x and y, 
achieving SOTA on WFLW dataset. 
Like object detection, heatmap predicts which point is a positive sample on the featuremap, 
represented as a highlighted area, while x and y offsets are responsible for predicting the specific coordinates of these positive samples.
And it achieves ** NME 3.95 on WFLW ** with no extern data.
Contact me if u have problem about it. 2120140200@mail.nankai.edu.cn :)

demo pictures:

![samples](https://github.com/610265158/face_landmark/blob/master/figures/tmp_screenshot_18.08.20192.png)

![gifs](https://github.com/610265158/Peppa_Pig_Face_Engine/blob/master/figure/sample.gif)

this gif is from github.com/610265158/Peppa_Pig_Face_Engine )



## metric

WLFW test set.

| WFLW    | inputsize | NME      | Flops(G) | Params(M) | Pose | Exp. | Ill. | Mu.  | Occ. | Blur | pretrained                                                                                      |
|---------|-----------|----------|----------|-----------|------|------|------|------|------|------|-------------------------------------------------------------------------------------------------|
| Student | 128x128   | **4.80** | 0.35     | 3.25      | 8.53 | 5.00 | 4.61 | 4.81 | 5.80 | 5.36 | [skps](https://drive.google.com/drive/folders/1JktGIKohpeLO14a6eJqNlZort_46qVC0?usp=share_link) |
| Teacher | 128x128   | **4.17** | 1.38     | 11.53     | 7.14 | 4.32 | 4.01 | 4.03 | 4.98 | 4.68 | [skps](https://drive.google.com/drive/folders/1JktGIKohpeLO14a6eJqNlZort_46qVC0?usp=share_link) |
| Student | 256x256   | **4.35** | 1.39     | 3.25      | 7.53 | 4.52 | 4.16 | 4.21 | 5.34 | 4.93 | [skps](https://drive.google.com/drive/folders/1Y8FvJV1X5YTUkwt5MywVFvqzStpxRK_S?usp=sharing)    |
| Teacher | 256x256   | **3.95** | 5.53     | 11.53     | 7.00 | 4.00 | 3.81 | 3.78 | 4.85 | 4.54 | [skps](https://drive.google.com/drive/folders/1Y8FvJV1X5YTUkwt5MywVFvqzStpxRK_S?usp=sharing)    |



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

### Evaluation


```
python tools/eval_WFLW.py --weight xxx.pth --data_dir ./ --img_size 256
```

```
python vis.py --model ./keypoints.pth
```
### visualization

```
python vis.py --model ./keypoints.pth
```



### convert to onyx

``` python
python tools/convert_to_onnx.py --model ./keypoints.pth
```



