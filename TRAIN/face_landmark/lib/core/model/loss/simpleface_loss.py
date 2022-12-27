# -*-coding:utf-8-*-
import sys


import torch
import math
import numpy as np

from train_config import config as cfg

bce_losser=torch.nn.BCEWithLogitsLoss()
def _wing_loss(landmarks, labels, w=10.0, epsilon=2.0, weights=1.):
    """
    Arguments:
        landmarks, labels: float tensors with shape [batch_size, landmarks].  landmarks means x1,x2,x3,x4...y1,y2,y3,y4   1-D
        w, epsilon: a float numbers.
    Returns:
        a float tensor with shape [].
    """

    x = landmarks - labels
    c = w * (1.0 - math.log(1.0 + w / epsilon))
    absolute_x = torch.abs(x)
    losses = torch.where(
        torch.gt(absolute_x, w), absolute_x - c,
        w * torch.log(1.0 + absolute_x / epsilon)

    )
    losses = losses * torch.tensor(cfg.DATA.weights,device='cuda')
    loss = torch.sum(torch.mean(losses * weights, dim=[0]))

    return loss


def _mse(landmarks, labels, weights=1.):
    return torch.mean(0.5 * (landmarks - labels) *(landmarks - labels)* weights)


def l1(landmarks, labels):
    return torch.mean(landmarks - labels)


def calculate_loss(predict_keypoints, label_keypoints):
    
    landmark_label = label_keypoints[:, 0:136]
    pose_label = label_keypoints[:, 136:139]
    leye_cls_label = label_keypoints[:, 139]
    reye_cls_label = label_keypoints[:, 140]
    mouth_cls_label = label_keypoints[:, 141]
    big_mouth_cls_label = label_keypoints[:, 142]

    landmark_predict = predict_keypoints[:, 0:136]
    pose_predict = predict_keypoints[:, 136:139]
    leye_cls_predict = predict_keypoints[:, 139]
    reye_cls_predict = predict_keypoints[:, 140]
    mouth_cls_predict = predict_keypoints[:, 141]
    big_mouth_cls_predict = predict_keypoints[:, 142]

    loss = _wing_loss(landmark_predict, landmark_label)

    loss_pose = _mse(pose_predict, pose_label)

    leye_loss = bce_losser(leye_cls_predict,leye_cls_label)
    reye_loss = bce_losser(reye_cls_predict,reye_cls_label)

    mouth_loss = bce_losser(mouth_cls_predict,mouth_cls_label)
    mouth_loss_big = bce_losser(big_mouth_cls_predict,big_mouth_cls_label)
    mouth_loss = mouth_loss + mouth_loss_big

    # ##make crosssentropy
    # leye_cla_correct_prediction = tf.equal(
    #     tf.cast(tf.greater_equal(tf.nn.sigmoid(leye_cls_predict), 0.5), tf.int32),
    #     tf.cast(leye_cla_label, tf.int32))
    # leye_cla_accuracy = tf.reduce_mean(tf.cast(leye_cla_correct_prediction, tf.float32))
    #
    # reye_cla_correct_prediction = tf.equal(
    #     tf.cast(tf.greater_equal(tf.nn.sigmoid(reye_cla_predict), 0.5), tf.int32),
    #     tf.cast(reye_cla_label, tf.int32))
    # reye_cla_accuracy = tf.reduce_mean(tf.cast(reye_cla_correct_prediction, tf.float32))
    # mouth_cla_correct_prediction = tf.equal(
    #     tf.cast(tf.greater_equal(tf.nn.sigmoid(mouth_cla_predict), 0.5), tf.int32),
    #     tf.cast(mouth_cla_label, tf.int32))
    # mouth_cla_accuracy = tf.reduce_mean(tf.cast(mouth_cla_correct_prediction, tf.float32))

    #### l2 regularization_losses
    # l2_loss = []
    # l2_reg = tf.keras.regularizers.l2(cfg.TRAIN.weight_decay_factor)
    # variables_restore = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
    # for var in variables_restore:
    #     if 'weight' in var.name:
    #         l2_loss.append(l2_reg(var))
    # regularization_losses = tf.add_n(l2_loss, name='l1_loss')

    return loss + loss_pose + leye_loss + reye_loss + mouth_loss


