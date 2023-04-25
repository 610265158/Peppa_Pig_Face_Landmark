import sys
sys.path.append('.')
import numpy as np
import cv2
import os
import math
import time
import  torch
import  MNN
import math
from config import config as cfg

from lib.core.api.onnx_model_base import ONNXEngine

class FaceDetector:
    def __init__(self,model_path='pretrained/yolov5n-0.5.onnx'):


        self.model=ONNXEngine(model_path)
        self.input_size=(384,640)

    def __call__(self, image,
                 score_threshold=cfg.DETECT.thres,
                 iou_threshold=cfg.DETECT.iou_thres):
        img_for_net, recover_info = self.preprocess(image)

        # Inference
        t0=time.time()
        output = self.model(img_for_net)
        print(time.time()-t0)
        output = np.reshape(output, ( 15120, 16))

        output[:,:4] = self.xywh2xyxy(output[:, :4])

        bboxes=self.py_nms(output,iou_threshold,score_threshold)

        bboxes[:, :4] = self.scale_coords(bboxes[:, :4], recover_info)


        return bboxes


    def preprocess(self, image, color=(114, 114, 114)):

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, c = img.shape  # orig hw

        scale = min(self.input_size[0] / h, self.input_size[1] / w)  # resize image to img_size

        img = cv2.resize(img, (int(w * scale), int(h * scale)))

        h, w, c = img.shape

        dh = (self.input_size[0] - h) / 2
        dw = (self.input_size[1] - w) / 2
        ##letter box

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        img = img.transpose(2, 0, 1).astype(np.float32)

        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        img = np.expand_dims(img, axis=0)

        return img, [scale, left, top]

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def scale_coords(self, bbox, revocer_info):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        [scale, dx, dy] = revocer_info

        bbox[:, 0] -= dx
        bbox[:, 1] -= dy
        bbox[:, 2] -= dx
        bbox[:, 3] -= dy

        bbox /= scale

        return bbox

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




if __name__ == "__main__":
    import  os

    model=FaceDetector('./pretrained/yolov5n-0.5.onnx')
    tes_dir='./figure'
    listdir = os.listdir(tes_dir)
    listdir=[x for x in listdir if 'jpg' in x]

    for file_path in listdir:
        img_path = os.path.join(tes_dir, file_path)
        image_ori = cv2.imread(img_path)
        image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)

        boxes=model(image)


        for i in range(boxes.shape[0]):
            box = boxes[i, :4].astype(np.int)

            score=boxes[i,4]
            cv2.rectangle(image_ori, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        cv2.imshow("UltraFace_mnn_py", image_ori)
        cv2.waitKey(-1)
