

import os
import numpy as np
from easydict import EasyDict as edict

config = edict()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"           ### if to use cuda,

config.DETECT = edict()
config.DETECT.model_path='./model/detector'         ### saved_model or tflite
config.DETECT.topk=10                               ###max boxes
config.DETECT.thres=0.5                             ###thres for nms
config.DETECT.iou_thres=0.3                         ###iou thres for nms
config.DETECT.input_shape=(256,320,3)               ###input shape for detector


config.KEYPOINTS = edict()
config.KEYPOINTS.model_path='./model/keypoints'     ### saved_model or tflite
config.KEYPOINTS.dense_dim=136+3+4                  #### output dimension
config.KEYPOINTS.p_num=68                           #### 68 points
config.KEYPOINTS.base_extend_range=[0.2,0.3]        ####
config.KEYPOINTS.input_shape = (160,160,3)          # input size during training , 160

config.TRACE= edict()
config.TRACE.ema_or_one_euro='euro'                 ### post process
config.TRACE.pixel_thres=1
config.TRACE.smooth_box=0.3                  ## if use euro, this will be disable
config.TRACE.smooth_landmark=0.95            ## if use euro, this will be disable
config.TRACE.iou_thres=0.5

config.DATA = edict()
config.DATA.pixel_means = np.array([127., 127., 127.]) # RGB








