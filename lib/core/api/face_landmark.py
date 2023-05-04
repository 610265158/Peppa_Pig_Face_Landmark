# -*-coding:utf-8-*-

import cv2
import numpy as np

from config import config as cfg
from lib.core.api.onnx_model_base import ONNXEngine



class FaceLandmark:
    """
        the model was constructed by the params in config.py
    """

    def __init__(self,model_path='pretrained/kps.MNN'):
        self.model=ONNXEngine(model_path)
        self.min_face = 20
        self.keypoints_num = cfg.KEYPOINTS.p_num

        self.input_size=(128,128)

    ##below are the method  run for one by one, will be deprecated in the future
    def __call__(self, img, bboxes):


        landmark_result = []
        states_result = []

        for i, bbox in enumerate(bboxes):

            image_croped, detail = self.preprocess(img, bbox, i)

            image_croped = image_croped.transpose((2, 0, 1)).astype(np.float32)

            image_croped=image_croped/255.
            image_croped=np.expand_dims(image_croped,axis=0)
            landmark,score=self.model(image_croped)


            state=score.reshape(-1)
            landmark=np.array(landmark)[:98*2].reshape(-1,2)


            landmark = self.postprocess(landmark, detail)

            if landmark is not None:
                landmark_result.append(landmark)
                states_result.append(state)

        return np.array(landmark_result), np.array(states_result)

    def preprocess(self, image, bbox, i):
        """
        :param image: raw image
        :param bbox: the bbox for the face
        :param i: index of face
        :return:
        """
        ##preprocess
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        if bbox_width <= self.min_face or bbox_height <= self.min_face:
            return None, None
        add = int(max(bbox_width, bbox_height))
        bimg = cv2.copyMakeBorder(image, add, add, add, add,
                                  borderType=cv2.BORDER_CONSTANT)
        bbox += add

        face_width = (1 + 2 * cfg.KEYPOINTS.base_extend_range[0]) * bbox_width
        center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]

        ### make the box as square
        bbox[0] = center[0] - face_width // 2
        bbox[1] = center[1] - face_width // 2
        bbox[2] = center[0] + face_width // 2
        bbox[3] = center[1] + face_width // 2

        # crop
        bbox = bbox.astype(np.int)
        crop_image = bimg[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

        h, w, _ = crop_image.shape
        crop_image = cv2.resize(crop_image, (cfg.KEYPOINTS.input_shape[1],
                                             cfg.KEYPOINTS.input_shape[0]))



        cv2.imshow('i am watching u * * %d' % i, crop_image)

        return crop_image, [h, w, bbox[1], bbox[0], add]

    def postprocess(self, landmark, detail):

        ##recorver, and grouped as [68,2]

        # landmark[:, 0] = landmark[:, 0] * w + bbox[0] -add
        # landmark[:, 1] = landmark[:, 1] * h + bbox[1] -add
        landmark[:, 0] = landmark[:, 0] * detail[1] + detail[3] - detail[4]
        landmark[:, 1] = landmark[:, 1] * detail[0] + detail[2] - detail[4]

        return landmark



    ##below are the method run for batch
