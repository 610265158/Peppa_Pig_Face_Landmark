import sys
sys.path.append('.')
import numpy as np
import cv2

import math
import time
import  torch
import  MNN
import math
from config import config as cfg


def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    """Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w).
    The conversion:
        $$predicted\_center * center_variance = \frac {real\_center - prior\_center} {prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac {real\_hw} {prior\_hw}$$
    We do it in the inverse direction here.
    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[center_x, center_y, h, w]]. All the values
            are relative to the image size.
    """
    # priors can have one dimension less.


    if len(priors.shape) + 1 == len(locations.shape):
        priors = np.expand_dims(priors, 0)
    return np.concatenate([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], axis=len(locations.shape) - 1)
def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    # priors can have one dimension less
    if len(center_form_priors.shape) + 1 == len(center_form_boxes.shape):
        center_form_priors = np.expand_dims(center_form_priors, 0)
    return np.concatenate([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
        np.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], axis=len(center_form_boxes.shape) - 1)


def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def center_form_to_corner_form(locations):
    return np.concatenate([locations[..., :2] - locations[..., 2:] / 2,
                           locations[..., :2] + locations[..., 2:] / 2], len(locations.shape) - 1)


def corner_form_to_center_form(boxes):
    return np.concatenate([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]
    ], len(boxes.shape) - 1)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    # _, indexes = scores.sort(descending=True)
    indexes = np.argsort(scores)
    # indexes = indexes[:candidate_size]
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        # current = indexes[0]
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        # indexes = indexes[1:]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

class FaceDetector:
    def __init__(self,mnn_model_path='pretrained/RFB-320.mnn'):

        self.image_mean = np.array([127, 127, 127])
        self.image_std = 128.0
        self.iou_threshold = 0.3
        self.threshold=0.3
        self.center_variance = 0.1
        self.size_variance = 0.2
        self.min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
        self.strides = [8, 16, 32, 64]

        self.interpreter = MNN.Interpreter(mnn_model_path)
        self.session = self.interpreter.createSession()
        self.input_tensor = self.interpreter.getSessionInput(self.session)

        self.input_size=(320,240)
        self.priors = self.define_img_size(self.input_size)



    def define_img_size(self,image_size):
        shrinkage_list = []
        feature_map_w_h_list = []
        for size in image_size:
            feature_map = [np.ceil(size / stride) for stride in self.strides]
            feature_map_w_h_list.append(feature_map)

        for i in range(0, len(image_size)):
            shrinkage_list.append(self.strides)
        priors = self.generate_priors(feature_map_w_h_list, shrinkage_list, image_size, self.min_boxes)

        return priors
    def generate_priors(self,feature_map_list, shrinkage_list, image_size, min_boxes, clamp=True):
            priors = []
            for index in range(0, len(feature_map_list[0])):
                scale_w = image_size[0] / shrinkage_list[0][index]
                scale_h = image_size[1] / shrinkage_list[1][index]
                for j in range(0, int(feature_map_list[1][index])):
                    for i in range(0, int(feature_map_list[0][index])):
                        x_center = (i + 0.5) / scale_w
                        y_center = (j + 0.5) / scale_h

                        for min_box in min_boxes[index]:
                            w = min_box / image_size[0]
                            h = min_box / image_size[1]
                            priors.append([
                                x_center,
                                y_center,
                                w,
                                h
                            ])
            print("priors nums:{}".format(len(priors)))
            priors = torch.tensor(priors)
            if clamp:
                torch.clamp(priors, 0.0, 1.0, out=priors)
            return priors

    def predict(self,width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = hard_nms(box_probs,
                                           iou_threshold=iou_threshold,
                                           top_k=top_k,
                                           )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

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


        ### w h
        input_size = self.input_size

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_fornet, scale_x, scale_y, dx, dy = self.preprocess(image,
                                                                 target_height=input_size[1],
                                                                 target_width=input_size[0])
        image_fornet = image_fornet.astype(np.float32)
        image_fornet = (image_fornet - self.image_mean) / self.image_std

        image_fornet = image_fornet.transpose((2, 0, 1))

        image_fornet = image_fornet.astype(np.float32)



        tmp_input = MNN.Tensor((1, 3, input_size[1], input_size[0]), MNN.Halide_Type_Float, image_fornet,
                               MNN.Tensor_DimensionType_Caffe)

        self.input_tensor.copyFrom(tmp_input)
        time_time = time.time()
        self.interpreter.runSession(self.session)
        scores = self.interpreter.getSessionOutput(self.session, "scores").getData()
        boxes = self.interpreter.getSessionOutput(self.session, "boxes").getData()
        boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
        scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
        print("inference time: {} s".format(round(time.time() - time_time, 4)))

        boxes = convert_locations_to_boxes(boxes, self.priors, self.center_variance, self.size_variance)
        boxes = center_form_to_corner_form(boxes)



        bboxes=np.concatenate([boxes,scores[:,:,1:2]],axis=-1)

        bboxes=self.py_nms(bboxes[0],iou_thres=iou_threshold,score_thres=score_threshold)

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

        bimage = np.zeros(shape=[target_height, target_width, c], dtype=image.dtype)
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




if __name__ == "__main__":
    import  os

    model=FaceDetector('./pretrained/RFB-320.mnn')
    tes_dir='./figure'
    listdir = os.listdir(tes_dir)
    listdir=[x for x in listdir if 'jpg' in x]

    for file_path in listdir:
        img_path = os.path.join(tes_dir, file_path)
        image_ori = cv2.imread(img_path)
        image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)

        boxes=model(image)


        print(boxes.shape)
        for i in range(boxes.shape[0]):
            box = boxes[i, :4].astype(np.int)

            score=boxes[i,4]
            cv2.rectangle(image_ori, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        cv2.imshow("UltraFace_mnn_py", image_ori)
        cv2.waitKey(-1)
