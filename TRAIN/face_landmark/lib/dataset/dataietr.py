import random
import cv2
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from lib.utils.logger import logger

import traceback

from train_config import config as cfg
import albumentations as A
import os
import copy
from lib.dataset.augmentor.augmentation import Rotate_aug, \
    Affine_aug, \
    Mirror, \
    Padding_aug

from lib.dataset.headpose import get_head_pose

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class AlaskaDataIter():
    def __init__(self, df, img_root,
                 training_flag=True, shuffle=True):
        self.eye_close_thres = 0.03
        self.mouth_close_thres = 0.02
        self.big_mouth_open_thres = 0.08

        self.training_flag = training_flag
        self.shuffle = shuffle
        self.img_root_path = img_root

        self.df = df

        # if self.training_flag:
        #     self.balance()

        self.train_trans = A.Compose([

            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.GaussianBlur(p=0.5),
            A.ToGray(p=0.1),
            A.GaussNoise(p=0.5),
            A.CoarseDropout(max_holes=8, max_width=16,
                            max_height=16,
                            p=0.5)

        ])

        self.val_trans = A.Compose([

            A.Resize(height=cfg.MODEL.hin,
                     width=cfg.MODEL.win)

        ])

    def __getitem__(self, item):

        return self.single_map_func(self.df[item], self.training_flag)

    def __len__(self):

        return len(self.df)

    def augmentationCropImage(self, img, bbox, joints=None, is_training=True):

        bbox = np.array(bbox).reshape(4, ).astype(np.float32)

        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]

        add = int(max(bbox_width, bbox_height))

        bimg = cv2.copyMakeBorder(img, add, add, add, add, borderType=cv2.BORDER_CONSTANT)

        objcenter = np.array([(bbox[0] + bbox[2]) / 2., (bbox[1] + bbox[3]) / 2.])
        bbox += add
        objcenter += add

        joints[:, :2] += add

        gt_width = (bbox[2] - bbox[0])
        gt_height = (bbox[3] - bbox[1])

        crop_width_half = gt_width * (1 + cfg.DATA.base_extend_range[0] * 2) // 2
        crop_height_half = gt_height * (1 + cfg.DATA.base_extend_range[1] * 2) // 2

        if is_training:
            min_x = int(objcenter[0] - crop_width_half + \
                        random.uniform(-cfg.DATA.base_extend_range[0], cfg.DATA.base_extend_range[0]) * gt_width)
            max_x = int(objcenter[0] + crop_width_half + \
                        random.uniform(-cfg.DATA.base_extend_range[0], cfg.DATA.base_extend_range[0]) * gt_width)
            min_y = int(objcenter[1] - crop_height_half + \
                        random.uniform(-cfg.DATA.base_extend_range[1], cfg.DATA.base_extend_range[1]) * gt_height)
            max_y = int(objcenter[1] + crop_height_half + \
                        random.uniform(-cfg.DATA.base_extend_range[1], cfg.DATA.base_extend_range[1]) * gt_height)
        else:
            min_x = int(objcenter[0] - crop_width_half)
            max_x = int(objcenter[0] + crop_width_half)
            min_y = int(objcenter[1] - crop_height_half)
            max_y = int(objcenter[1] + crop_height_half)

        joints[:, 0] = joints[:, 0] - min_x
        joints[:, 1] = joints[:, 1] - min_y

        img = bimg[min_y:max_y, min_x:max_x, :]

        crop_image_height, crop_image_width, _ = img.shape
        joints[:, 0] = joints[:, 0] / crop_image_width
        joints[:, 1] = joints[:, 1] / crop_image_height

        # interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST,
        #                   cv2.INTER_LANCZOS4]
        # interp_method = random.choice(interp_methods)

        img = cv2.resize(img, (cfg.MODEL.win, cfg.MODEL.hin))

        joints[:, 0] = joints[:, 0] * cfg.MODEL.win
        joints[:, 1] = joints[:, 1] * cfg.MODEL.hin
        return img, joints

    def gaussian_k(self, x0, y0, sigma, width, height):
        """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
        """
        x = np.arange(0, width, 1, float)  ## (width,)
        y = np.arange(0, height, 1, float)[:, np.newaxis]  ## (height,1)
        return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def generate_hm(self, height, width, landmarks, s=3):
        """ Generate a full Heap Map for every landmarks in an array
        Args:
            height    : The height of Heat Map (the height of target output)
            width     : The width  of Heat Map (the width of target output)
            joints    : [(x1,y1),(x2,y2)...] containing landmarks
            maxlenght : Lenght of the Bounding Box
        """
        Nlandmarks = len(landmarks)
        hm = np.zeros((height, width, Nlandmarks), dtype=np.float32)
        # if not np.array_equal(landmarks[i], [-1, -1]):
        floored_landmarks = np.round(landmarks)

        for i in range(Nlandmarks):
            # if not np.array_equal(landmarks[i], [-1, -1]):

            hm[:, :, i] = self.gaussian_k(floored_landmarks[i][0],
                                          floored_landmarks[i][1],
                                          s, height, width)


        ### make offside


        offside_y, offside_x = np.meshgrid(np.arange(0, height), np.arange(0, width), indexing='ij')
        offside_y=np.expand_dims(offside_y,axis=-1)
        offside_y=np.repeat(offside_y,Nlandmarks,axis=-1)

        offside_x = np.expand_dims(offside_x, axis=-1)
        offside_x = np.repeat(offside_x, Nlandmarks, axis=-1)

        offside_x = landmarks[:, 0] - offside_x
        offside_y = landmarks[:, 1] - offside_y

        hm=np.concatenate([hm,offside_x,offside_y],axis=-1)
        return hm


    def single_map_func(self, dp, is_training):
        """Data augmentation function."""
        ####customed here

        if 'wink' in dp:
            dp = dp.split()
            kps = dp[:98 * 2]
            fn = dp[-1]
            image = cv2.imread(fn)

        else:
            dp = dp.split()
            kps = dp[:98 * 2]
            fn = dp[-1]
            image = cv2.imread(os.path.join(self.img_root_path, fn))

        kps = np.array(kps, dtype=np.float32).reshape([-1, 2])
        bbox = [float(np.min(kps[:, 0])), float(np.min(kps[:, 1])), float(np.max(kps[:, 0])),
                float(np.max(kps[:, 1]))]

        bbox = np.array(bbox)

        ### random crop and resize
        crop_image, label = self.augmentationCropImage(image, bbox, kps, is_training)

        if is_training:

            if random.uniform(0, 1) > 0.5:
                crop_image, label = Mirror(crop_image, label=label, symmetry=cfg.DATA.symmetry)
            if random.uniform(0, 1) > 0.5:
                angle = random.uniform(-30, 30)
                crop_image, label = Rotate_aug(crop_image, label=label, angle=angle)

            if random.uniform(0, 1) > 0.7:
                strength = random.uniform(0, 50)
                crop_image, label = Affine_aug(crop_image, strength=strength, label=label)

            if random.uniform(0, 1) > 0.7:
                crop_image = Padding_aug(crop_image, 0.3)

            transformed = self.train_trans(image=crop_image)
            crop_image = transformed['image']

        #######head pose
        reprojectdst, euler_angle = get_head_pose(label, crop_image)
        PRY = euler_angle.reshape([-1]).astype(np.float32) / 90.

        ######cla_label
        cls_label = np.zeros([4])
        if np.sqrt(np.square(label[62, 0] - label[66, 0]) +
                   np.square(label[62, 1] - label[66, 1])) / cfg.MODEL.hin < self.eye_close_thres:
            cls_label[0] = 1

        if np.sqrt(np.square(label[70, 0] - label[74, 0]) +
                   np.square(label[70, 1] - label[74, 1])) / cfg.MODEL.hin < self.eye_close_thres:
            cls_label[1] = 1

        if np.sqrt(np.square(label[89, 0] - label[95, 0]) +
                   np.square(label[89, 1] - label[95, 1])) / cfg.MODEL.hin < self.mouth_close_thres \
                or np.sqrt(np.square(label[90, 0] - label[94, 0]) +
                           np.square(label[90, 1] - label[94, 1])) / cfg.MODEL.hin < self.mouth_close_thres \
                or np.sqrt(np.square(label[91, 0] - label[93, 0]) +
                           np.square(label[91, 1] - label[93, 1])) / cfg.MODEL.hin < self.mouth_close_thres:
            cls_label[2] = 1

        ### mouth open big   1 mean true
        if np.sqrt(np.square(label[90, 0] - label[94, 0]) +
                   np.square(label[90, 1] - label[94, 1])) / cfg.MODEL.hin > self.big_mouth_open_thres:
            cls_label[3] = 1


        kps_weight = np.ones_like(label)
        cls_weight = np.ones_like(cls_label)

        crop_image_height, crop_image_width, _ = crop_image.shape

        label = label.astype(np.float32)

        label[:, 0] = label[:, 0] / crop_image_width
        label[:, 1] = label[:, 1] / crop_image_height
        crop_image = crop_image.astype(np.float32)

        crop_image = np.transpose(crop_image, axes=[2, 0, 1])
        crop_image /= 255.

        label = label.reshape([-1]).astype(np.float32)
        kps_weight = kps_weight.reshape([-1]).astype(np.float32)
        cls_label = cls_label.astype(np.float32)
        cls_weight = cls_weight.astype(np.float32)

        total_label = np.concatenate([label, PRY, cls_label, kps_weight, cls_weight], axis=0)

        ### hm size inputsize//4

        kps = label.reshape([-1, 2])
        kps[:, 0] *= cfg.MODEL.win // 4
        kps[:, 1] *= cfg.MODEL.hin // 4

        hm = self.generate_hm(cfg.MODEL.hin // 4, cfg.MODEL.win // 4, kps, 2)

        hm = np.transpose(hm, axes=[2, 0, 1])

        return fn, crop_image, total_label, hm
