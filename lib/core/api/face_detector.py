import numpy as np
import cv2
import tensorflow as tf

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


    def __call__(self, image):
        """Detect faces.

        Arguments:
            image: a numpy uint8 array with shape [height, width, 3],
                that represents a RGB image.
        Returns:
            boxes: a float numpy array of shape [num_faces, 5].

        """

        image, scale_x, scale_y = self.preprocess(image,
                                                  target_width=self.input_shape[1],
                                                  target_height=self.input_shape[0])

        image = np.expand_dims(image, 0)
        res = self.model.inference(image)

        boxes = res['boxes'].numpy()
        scores = res['scores'].numpy()
        num_boxes = res['num_boxes'].numpy()



        ##sqeeze the box
        num_boxes = num_boxes[0]
        boxes = boxes[0][:num_boxes]
        scores = scores[0][:num_boxes]

        to_keep = scores > self.thres
        boxes = boxes[to_keep]
        scores = scores[to_keep]

        ###recorver to raw image
        scaler = np.array([self.input_shape[0] / scale_y,
                           self.input_shape[1] / scale_x,
                           self.input_shape[0] / scale_y,
                           self.input_shape[1] / scale_x], dtype='float32')
        boxes = boxes * scaler

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
        long_side = max(h, w)

        scale_x = scale_y = target_height / long_side

        image_resized = cv2.resize(image, None, fx=scale_x, fy=scale_y)

        h_resized, w_resized, _ = image_resized.shape
        bimage[:h_resized, :w_resized, :] = image_resized

        return bimage, scale_x, scale_y



