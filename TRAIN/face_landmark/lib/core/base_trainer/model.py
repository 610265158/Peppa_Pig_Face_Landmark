import math
from functools import partial
import sys


import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from torchvision.models.mobilenetv3 import InvertedResidual,InvertedResidualConfig



bn_momentum=0.1



def upsample_x_like_y(x,y):
    size = y.shape[-2:]
    x=F.interpolate(x, size=size, mode='bilinear')

    return x

# from lib.core.base_trainer.mobileone import MobileOneBlock
class SeparableConv2d(nn.Module):
    """ Separable Conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding=0, bias=False,
                 channel_multiplier=1., pw_kernel_size=1):
        super(SeparableConv2d, self).__init__()


        self.conv_dw = nn.Sequential(nn.Conv2d(
            int(in_channels*channel_multiplier), int(in_channels*channel_multiplier), kernel_size,
            stride=stride, dilation=dilation, padding=padding, groups=int(in_channels*channel_multiplier)),
            nn.BatchNorm2d(in_channels,momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.conv_pw = nn.Conv2d(
            int(in_channels*channel_multiplier), out_channels, pw_kernel_size, padding=0, bias=bias)

    @property
    def in_channels(self):
        return self.conv_dw.in_channels

    @property
    def out_channels(self):
        return self.conv_pw.out_channels

    def forward(self, x):

        x = self.conv_dw(x)
        x = self.conv_pw(x)
        return x



class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.pool=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels,momentum=bn_momentum),
            nn.ReLU())
    def forward(self, x):

        y=x


        x = self.pool(x)

        x= upsample_x_like_y(x,y)
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 512

        rate1, rate2, rate3 = tuple(atrous_rates)

        self.fm_conx1=nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 1, bias=False),
            nn.BatchNorm2d(out_channels//4,momentum=bn_momentum),
            nn.ReLU())

        self.fm_convx3_rate2=nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=2, bias=False,dilation=rate1),
            nn.BatchNorm2d(out_channels//4,momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )

        self.fm_convx3_rate4=nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=4, bias=False,dilation=rate2),
            nn.BatchNorm2d(out_channels//4,momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )

        self.fm_convx3_rate8=nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=8, bias=False,dilation=rate3),
            nn.BatchNorm2d(out_channels//4,momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )

        self.fm_pool=ASPPPooling(in_channels=in_channels,out_channels=out_channels//4)

        self.project = nn.Sequential(
            nn.Conv2d(out_channels//4*5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels,momentum=bn_momentum),
            nn.ReLU(inplace=True))

    def forward(self, x):

        fm1=self.fm_conx1(x)
        fm2=self.fm_convx3_rate2(x)
        fm4=self.fm_convx3_rate4(x)
        fm8=self.fm_convx3_rate8(x)
        fm_pool=self.fm_pool(x)

        res = torch.cat([fm1,fm2,fm4,fm8,fm_pool], dim=1)

        return self.project(res)



class FeatureFuc(nn.Module):
    def __init__(self, inchannels=128):
        super(FeatureFuc, self).__init__()


        self.block1=InvertedResidual(cnf=InvertedResidualConfig( inchannels, 5,  256,  inchannels, False, "RE", 1, 1, 1),
                                     norm_layer = partial(nn.BatchNorm2d,  momentum=bn_momentum))

        self.block2=InvertedResidual(cnf=InvertedResidualConfig( inchannels, 5,  256,  inchannels, False, "RE", 1, 1, 1),
                                     norm_layer = partial(nn.BatchNorm2d,  momentum=bn_momentum))


    def forward(self, x):

        y1=self.block1(x)

        y2=self.block2(y1)

        return y2

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x  *self.cSE(x) +  x*self.sSE(x)






class Heading(nn.Module):

    def __init__(self,encoder_channels):
        super(Heading, self).__init__()
        self.extra_feature = FeatureFuc(encoder_channels[-1])
        self.aspp = ASPP(encoder_channels[-1], [2, 4, 8])






    def forward(self,features):

        img,encx2,encx4,encx8,encx16=features

        ## add extra feature
        encx16=self.extra_feature(encx16)
        encx16=self.aspp(encx16)

        return [encx2,encx4,encx8,encx16]



class Net(nn.Module):
    def __init__(self,):
        super(Net, self).__init__()

        self.encoder = timm.create_model(model_name='mobilenetv3_large_100',
                                         pretrained=True,
                                         features_only=True,
                                         out_indices=[0,1,2,4],
                                         bn_momentum=bn_momentum,
                                         in_chans=3,
                                         output_stride=16,
                                         )

        # self.encoder.blocks[4][1]=nn.Identity()
        self.encoder.blocks[5]=nn.Identity()
        self.encoder.blocks[6]=nn.Identity()


        self.encoder_out_channels = [3, 16, 24, 40, 112] #mobilenetv3


        self.head=Heading(self.encoder_out_channels)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)

        self._fc = nn.Linear(512, 136+3+4, bias=True)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        bs = x.size(0)
        features=self.encoder(x)

        features=[x]+features

        [encx2,encx4,encx8,encx16]=self.head(features)
        fm=self._avg_pooling(encx16)

        fm = fm.view(bs, -1)
        x = self._fc(fm)


        return x,[encx2,encx4,encx8,encx16]



class TeacherNet(nn.Module):
    def __init__(self,):
        super(TeacherNet, self).__init__()

        self.encoder = timm.create_model(model_name='tf_efficientnet_b0_ns',
                                         pretrained=True,
                                         features_only=True,
                                         out_indices=[0,1,2,3],
                                         bn_momentum=bn_momentum,
                                         in_chans=3,
                                         )

        # self.encoder.out_channels=[3, 24 , 40, 64,176]
        self.encoder.out_channels=[3,16, 24, 40, 112]
        self.head = Heading(self.encoder.out_channels)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)

        self._fc = nn.Linear(512, 136 + 3 + 4, bias=True)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        bs = x.size(0)
        features=self.encoder(x)

        features=[x]+features

        [encx2,encx4,encx8,encx16] = self.head(features)

        fm = self._avg_pooling(encx16)

        fm = fm.view(bs, -1)

        x = self._fc(fm)

        return x,[encx2,encx4,encx8,encx16]



class COTRAIN(nn.Module):
    def __init__(self,inference=False):
        super(COTRAIN, self).__init__()

        self.inference=inference
        self.student=Net()
        self.teacher=TeacherNet()

        self.MSELoss=nn.MSELoss()

        # self.DiceLoss    = segmentation_models_pytorch.losses.DiceLoss(mode='multilabel',eps=1e-5)
        self.BCELoss     = nn.BCEWithLogitsLoss()


        self.act=nn.Sigmoid()


    def distill_loss(self,student_pres,teacher_pres):


        num_level=len(student_pres)
        loss=0
        for i in range(num_level):
            loss+=self.MSELoss(student_pres[i],teacher_pres[i])

        return loss/num_level


    def criterion(self,y_pred, y_true):

        return 0.5*self.BCELoss(y_pred, y_true) + 0.5*self.DiceLoss(y_pred, y_true)

    def _wing_loss(self,landmarks, labels, w=10.0, epsilon=2.0, weights=1.):
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
        losses = losses * torch.tensor(cfg.DATA.weights, device='cuda')
        loss = torch.sum(torch.mean(losses * weights, dim=[0]))

        return loss
    def loss(self,predict_keypoints, label_keypoints):

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

        loss = self._wing_loss(landmark_predict, landmark_label)

        loss_pose = self.MSELoss(pose_predict, pose_label)

        leye_loss =  self.BCELoss  (leye_cls_predict, leye_cls_label)
        reye_loss =  self.BCELoss  (reye_cls_predict, reye_cls_label)

        mouth_loss =  self.BCELoss  (mouth_cls_predict, mouth_cls_label)
        mouth_loss_big =  self.BCELoss  (big_mouth_cls_predict, big_mouth_cls_label)
        mouth_loss = mouth_loss + mouth_loss_big



        return loss + loss_pose + leye_loss + reye_loss + mouth_loss

        return current_loss


    def forward(self, x,gt=None):

        student_pre,student_fms=self.student(x)

        if self.inference:

            return student_pre

        teacher_pre,teacher_fms=self.teacher(x)

        distill_loss=self.distill_loss(student_fms,teacher_fms)

        student_loss=self.loss(student_pre,gt)

        teacher_loss=self.loss(teacher_pre,gt)

        return student_loss,teacher_loss,distill_loss,student_pre








if __name__=='__main__':
    import torch
    import torchvision

    from thop import profile

    dummy_x = torch.randn(1, 3, 288, 160, device='cpu')

    model = COTRAIN(inference=True)

    input = torch.randn(1, 3, 288, 160)
    flops, params = profile(model, inputs=(input,))
    print(flops/1024/1024)

