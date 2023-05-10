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

| Model       | NME      | Flops(G) | Params(M) | Pose | Exp. | Ill. | Mu.  | Occ. | Blur | pretrained                                                                                      |
|-------------|----------|----------|-----------|------|------|------|------|------|------|-------------------------------------------------------------------------------------------------|
| Student@128 | **4.62** | 0.38     | 3.41      | 8.03 | 4.74 | 4.46 | 4.55 | 5.61 | 5.17 | [skps](https://drive.google.com/drive/folders/1qi3BfS-pJgMTrL5bPzcIJxh0pLxHrGxd?usp=sharing) |
| Teacher@128 | **4.02** | 1.44     | 11.23     | 6.84 | 4.10 | 3.93 | 3.90 | 4.84 | 4.56 | [skps](https://drive.google.com/drive/folders/1qi3BfS-pJgMTrL5bPzcIJxh0pLxHrGxd?usp=sharing) |
| Student@256 | **4.43** | 1.50     | 3.41      | 7.84 | 4.48 | 4.19 | 4.29 | 5.63 | 5.12 | [skps](https://drive.google.com/drive/folders/1qCdK5igHlSYMTxVxbH0XnZgACqzpz20R?usp=sharing)    |
| Teacher@256 | **3.90** | 5.76     | 11.23     | 6.67 | 3.94 | 3.78 | 3.78 | 4.82 | 4.52 | [skps](https://drive.google.com/drive/folders/1qCdK5igHlSYMTxVxbH0XnZgACqzpz20R?usp=sharing)    |



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



