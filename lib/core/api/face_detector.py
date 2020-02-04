import numpy as np
import cv2
import tensorflow as tf
import math
import time

from config import config as cfg

class FaceDetector:
    def __init__(self):
        """
        the model was constructed by the params in config.py
        """

        self.model_path=cfg.DETECT.model_path
        self.thres=cfg.DETECT.thres
        self.input_shape=cfg.DETECT.input_shape



        if 'lite' in self.model_path:
            self.model = tf.lite.Interpreter(model_path=self.model_path)
            self.model.allocate_tensors()
            self.input_details = self.model.get_input_details()
            self.output_details = self.model.get_output_details()
            self.tflite=True
        else:
            self.model = tf.saved_model.load(self.model_path)

            self.tflite = False


    def __call__(self, image,
                 score_threshold=cfg.DETECT.thres,
                 iou_threshold=cfg.DETECT.iou_thres,
                 input_shape=(240,320)):
        """Detect faces.

        Arguments:
            image: a numpy uint8 array with shape [height, width, 3],
                that represents a RGB image.
            input_shape: (h,w)
            score_threshold: a float number.
            iou_thres: a float number.
        Returns:
            boxes: a float numpy array of shape [num_faces, 5].

        """
        if not self.tflite:
            if input_shape is None:
                h,w,c=image.shape
                input_shape = (math.ceil(h / 64 ) * 64,
                               math.ceil(w / 64 ) * 64)
            else:
                h, w = input_shape
                input_shape = (math.ceil(h / 64 ) * 64,
                               math.ceil(w / 64 ) * 64)

            image_fornet, scale_x, scale_y,dx,dy = self.preprocess(image,
                                                                   target_height=input_shape[0],
                                                                   target_width =input_shape[1])

            if cfg.DETECT.input_shape[2] == 1:
                ##gray scale
                image_fornet = cv2.cvtColor(image_fornet, cv2.COLOR_RGB2GRAY)
                image_fornet = np.expand_dims(image_fornet, axis=-1)


            image_fornet = np.expand_dims(image_fornet, 0)

            start = time.time()
            bboxes = self.model.inference(image_fornet)
            print('xx', time.time() - start)
        else:
            input_shape = (320, 320)
            image_fornet, scale_x, scale_y, dx, dy = self.preprocess(image,
                                                                     target_height=input_shape[0],
                                                                     target_width=input_shape[1])
            if cfg.DETECT.input_shape[2] == 1:
                ##gray scale
                image_fornet = cv2.cvtColor(image_fornet, cv2.COLOR_RGB2GRAY)
                image_fornet = np.expand_dims(image_fornet, axis=-1)


            image_fornet = np.expand_dims(image_fornet, 0).astype(np.float32)

            start = time.time()
            self.model.set_tensor(self.input_details[0]['index'], image_fornet)

            self.model.invoke()

            bboxes = self.model.get_tensor(self.output_details[0]['index'])

            print('xx', time.time() - start)

        bboxes=self.py_nms(np.array(bboxes[0]),iou_thres=iou_threshold,score_thres=score_threshold)

        ###recorver to raw image
        boxes_scaler = np.array([  (input_shape[1]) / scale_x,
                                   (input_shape[0]) / scale_y,
                                   (input_shape[1]) / scale_x,
                                   (input_shape[0]) / scale_y,
                                    1.], dtype='float32')

        boxes_bias=np.array( [ dx / scale_x,
                               dy / scale_y,
                               dx / scale_x,
                               dy / scale_y,
                               0.], dtype='float32')
        bboxes = bboxes * boxes_scaler-boxes_bias

        return bboxes


    def preprocess(self, image, target_height, target_width):

        h, w, c = image.shape

        bimage = np.zeros(shape=[target_height, target_width, c], dtype=image.dtype) + np.array(cfg.DATA.pixel_means,
                                                                                                dtype=image.dtype)
        scale_y = target_height / h
        scale_x = target_width / w

        scale=min(scale_x,scale_y)

        image = cv2.resize(image, None, fx=scale, fy=scale)

        h_, w_, _ = image.shape

        dx=(target_width-w_)//2
        dy=(target_height-h_)//2
        bimage[dy:h_+dy, dx:w_+dx, :] = image

        return bimage, scale, scale, dx, dy

    def py_nms(self,bboxes, iou_thres, score_thres):

        upper_thres = np.where(bboxes[:, 4] > score_thres)[0]

        bboxes = bboxes[upper_thres]

        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        order = np.argsort(bboxes[:, 4])[::-1]

        keep = []

        while order.shape[0] > 0:
            cur = order[0]

            keep.append(cur)

            area = (bboxes[cur, 2] - bboxes[cur, 0]) * (bboxes[cur, 3] - bboxes[cur, 1])

            x1_reain = x1[order[1:]]
            y1_reain = y1[order[1:]]
            x2_reain = x2[order[1:]]
            y2_reain = y2[order[1:]]

            xx1 = np.maximum(bboxes[cur, 0], x1_reain)
            yy1 = np.maximum(bboxes[cur, 1], y1_reain)
            xx2 = np.minimum(bboxes[cur, 2], x2_reain)
            yy2 = np.minimum(bboxes[cur, 3], y2_reain)

            intersection = np.maximum(0, yy2 - yy1) * np.maximum(0, xx2 - xx1)

            iou = intersection / (area + (y2_reain - y1_reain) * (x2_reain - x1_reain) - intersection)

            ##keep the low iou
            low_iou_position = np.where(iou < iou_thres)[0]

            order = order[low_iou_position + 1]

        return bboxes[keep]




