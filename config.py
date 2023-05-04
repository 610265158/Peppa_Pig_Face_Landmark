

import os
import numpy as np
from easydict import EasyDict as edict

config = edict()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"           ### if to use cuda,

config.DETECT = edict()
config.DETECT.model_path='./pretrained/yolov5n-0.5.onnx'         ### saved_model or tflite
config.DETECT.topk=10                               ### max boxes
config.DETECT.min_face=1600                         ### max boxes
config.DETECT.thres=0.5                             ### thres for nms
config.DETECT.iou_thres=0.3                         ### iou thres for nms



config.KEYPOINTS = edict()
config.KEYPOINTS.model_path='./pretrained/kps_student.onnx'     ### saved_model or tflite
config.KEYPOINTS.dense_dim=136+3+4                  #### output dimension
config.KEYPOINTS.p_num=68                           #### 68 points
config.KEYPOINTS.base_extend_range=[0.2,0.3]        ####
config.KEYPOINTS.input_shape = (256,256,3)          # input size during training , 160

config.TRACE= edict()

config.TRACE.pixel_thres=3
config.TRACE.smooth_box=0.3                         ## if use euro, this will be disable
config.TRACE.iou_thres=0.5









config.http_server = edict()
config.http_server.ip="0.0.0.0"
config.http_server.port=5000


