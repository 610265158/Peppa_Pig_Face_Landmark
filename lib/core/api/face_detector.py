import numpy as np
import cv2
import tensorflow as tf
import math
import time

from config import config as cfg
from lib.logger.logger import logger


class FaceDetector:
    def __init__(self):
        """
        the model was constructed by the params in config.py
        """

        self.model_path=cfg.DETECT.model_path
        self.thres=cfg.DETECT.thres
        self.input_shape=cfg.DETECT.input_shape
        logger.info('INIT THE FACELANDMARK MODEL...')
        self.model =tf.saved_model.load(cfg.DETECT.model_path)

    def __call__(self,
                 image,
                 score_threshold=cfg.DETECT.thres,
                 input_shape=(cfg.DETECT.input_shape[0], cfg.DETECT.input_shape[1])):
        """Detect faces.

        Arguments:
            image: a numpy uint8 array with shape [height, width, 3],
                that represents a RGB image.
            input_shape: (h,w)
            score_threshold: a float number.
        Returns:
            boxes: a float numpy array of shape [num_faces, 5].

        """

        if input_shape is None:
            h, w, c = image.shape
            input_shape = (math.ceil(h / 32) * 32, math.ceil(w / 32) * 32)
        else:
            h, w = input_shape
            input_shape = (math.ceil(h / 32) * 32, math.ceil(w / 32) * 32)

        image_fornet, scale_x, scale_y, dx, dy = self.preprocess(image,
                                                                 target_height=input_shape[0],
                                                                 target_width=input_shape[1])

        image_fornet = np.expand_dims(image_fornet, 0)

        start = time.time()
        res = self.model.inference(image_fornet)

        print('xx', time.time() - start)

        boxes = res['boxes'].numpy()
        label = res['labels'].numpy()
        scores = res['scores'].numpy()
        num_boxes = res['num_boxes'].numpy()

        num_boxes = num_boxes[0]
        boxes = boxes[0][:num_boxes]
        label = label[0][:num_boxes]
        scores = scores[0][:num_boxes]

        to_keep = scores > score_threshold
        boxes = boxes[to_keep]
        scores = scores[to_keep]
        label = label[to_keep]
        ###recorver to raw image
        boxes_scaler = np.array([(input_shape[0]) / scale_y,
                                 (input_shape[1]) / scale_x,
                                 (input_shape[0]) / scale_y,
                                 (input_shape[1]) / scale_x], dtype='float32')

        boxes_bias = np.array([dy / scale_y,
                               dx / scale_x,
                               dy / scale_y,
                               dx / scale_x], dtype='float32')
        boxes = boxes * boxes_scaler - boxes_bias

        scores = np.expand_dims(scores, 0).reshape([-1, 1])

        #####the tf.nms produce ymin,xmin,ymax,xmax,  swap it in to xmin,ymin,xmax,ymax
        for i in range(boxes.shape[0]):
            boxes[i] = np.array([boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2]])
        return np.concatenate([boxes, scores], axis=1)

    def preprocess(self, image, target_height, target_width, label=None):

        ###sometimes use in objs detects
        h, w, c = image.shape

        bimage = np.zeros(shape=[target_height, target_width, c], dtype=image.dtype) + np.array(cfg.DATA.pixel_means,
                                                                                                dtype=image.dtype)

        scale_y = target_height / h
        scale_x = target_width / w

        scale = min(scale_x, scale_y)

        image = cv2.resize(image, None, fx=scale, fy=scale)

        h_, w_, _ = image.shape

        dx = (target_width - w_) // 2
        dy = (target_height - h_) // 2
        bimage[dy:h_ + dy, dx:w_ + dx, :] = image

        return bimage, scale, scale, dx, dy



