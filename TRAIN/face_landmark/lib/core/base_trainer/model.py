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
    def __init__(self, in_channels, atrous_rates,out_channels=512):
        super(ASPP, self).__init__()


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

        # self.fm_convx3_rate8=nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=8, bias=False,dilation=rate3),
        #     nn.BatchNorm2d(out_channels//4,momentum=bn_momentum),
        #     nn.ReLU(inplace=True)
        # )

        self.fm_pool=ASPPPooling(in_channels=in_channels,out_channels=out_channels//4)

        self.project = nn.Sequential(
            nn.Conv2d(out_channels//4*4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels,momentum=bn_momentum),
            nn.ReLU(inplace=True))

    def forward(self, x):

        fm1=self.fm_conx1(x)
        fm2=self.fm_convx3_rate2(x)
        fm4=self.fm_convx3_rate4(x)
        # fm8=self.fm_convx3_rate8(x)
        fm_pool=self.fm_pool(x)

        res = torch.cat([fm1,fm2,fm4,fm_pool], dim=1)

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


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_separable_conv=True,
            use_attention=False,
            use_second_conv=False,
            kernel_size=5,
    ):
        super().__init__()
        if use_separable_conv:
            self.conv1 = nn.Sequential(SeparableConv2d(
                in_channels+skip_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size//2,
                ),
                nn.BatchNorm2d(out_channels,momentum=bn_momentum),
                nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(
                in_channels+skip_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size//2,
                ),
                nn.BatchNorm2d(out_channels,momentum=bn_momentum),
                nn.ReLU(inplace=True))

        # self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        if use_second_conv:
            self.conv2 = nn.Sequential(nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1),
                nn.BatchNorm2d(out_channels,momentum=bn_momentum),
                nn.ReLU(inplace=True))
        else:
            self.conv2 = nn.Identity()

        if use_attention:
            self.attention2 = SCSEModule(in_channels=out_channels)
        else:
            self.attention2 =nn.Identity()

    def forward(self, x, skip=None):

        # x= upsample_x_like_y(x,skip)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        if skip is None:
            return x


        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            # x = self.attention1(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.attention2(x)
        return x


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Matting(nn.Module):

    def __init__(self,encoder_channels):
        super(Matting, self).__init__()
        self.extra_feature = FeatureFuc(encoder_channels[-1])
        self.aspp = ASPP(encoder_channels[-1], [2, 4, 8])

        self.upsampler1=DecoderBlock(512, encoder_channels[-2], 64, \
                                       use_separable_conv=True, \
                                       use_attention=True,
                                       kernel_size=5)

        self.upsampler2 = DecoderBlock(64, encoder_channels[-3], 32, \
                                       use_separable_conv=True, \
                                       use_attention=False,
                                       use_second_conv=True,
                                       kernel_size=5)

        self.upsampler3 = DecoderBlock(32, encoder_channels[-4], 32, \
                                       use_separable_conv=False, \
                                       use_attention=False,
                                       kernel_size=3)


        self.apply(weight_init)
    def forward(self,features):



        img,encx2,encx4,encx8,encx16=features

        ## add extra feature
        encx16=self.extra_feature(encx16)
        encx16=self.aspp(encx16)

        decx8=self.upsampler1(encx16,encx8)
        decx4 = self.upsampler2(decx8, encx4)
        decx2=self.upsampler3(decx4, encx2)

        #### semantic predict


        return[decx2,decx4,decx8,encx16]

class Net(nn.Module):
    def __init__(self,):
        super(Net, self).__init__()

        self.encoder = timm.create_model(model_name='mobilenetv3_large_100.ra_in1k',
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

        # print(self.encoder)


        # shuffle=shufflenet_v2_x1_0(weights=torchvision.models.ShuffleNet_V2_X1_0_Weights)
        #
        # nodes, _ = get_graph_node_names(shuffle)
        #
        # self.encoder = create_feature_extractor(
        #     shuffle, return_nodes=['conv1', 'maxpool', 'stage2', 'stage3'])
        #
        # # self.encoder=MobileNetV2Backbone(in_channels=3)
        # self.encoder_out_channels=[3, 24,24,116, 232 ]  ##shufflenet1.0
        self.encoder_out_channels = [3, 16, 24, 40, 112] #mobilenetv3
        # self.encoder.out_channels=[3,16, 24, 32, 88, 720]


        self.matting=Matting(self.encoder_out_channels)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(640, 98 * 2 + 3 + 4, bias=True)


        self.hm=nn.Conv2d(in_channels=32,out_channels=98,kernel_size=3,stride=1,padding=1,bias=True)


        weight_init(self.fc)
        weight_init(self.hm)
    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        bs=x.size(0)
        features=self.encoder(x)

        features=[x]+features

        [encx2,encx4,encx8,encx16]=self.matting(features)

        fmx16 = self._avg_pooling(encx16)
        fmx8 = self._avg_pooling(encx8)
        fmx4 = self._avg_pooling(encx4)
        fmx2 = self._avg_pooling(encx2)



        fm=torch.cat([fmx2,fmx4,fmx8,fmx16],dim=1)

        fm = fm.view(bs, -1)
        x = self.fc(fm)

        hm=self.hm(encx2)


        return x,hm,[encx2,encx4,encx8,encx16,x]



class TeacherNet(nn.Module):
    def __init__(self,):
        super(TeacherNet, self).__init__()

        self.encoder = timm.create_model(model_name='efficientnet_b5.in12k_ft_in1k',
                                         pretrained=True,
                                         features_only=True,
                                         out_indices=[0,1,2,3],
                                         bn_momentum=bn_momentum,
                                         in_chans=3,
                                         )
        # print(self.encoder)
        # print(self.encoder.blocks[6])

        # self.encoder.blocks[6]=torch.nn.Identity()
        # self.encoder=MobileNetV2Backbone(in_channels=3)
        self.encoder.out_channels=[3, 24 , 40, 64,176]
        # self.encoder.out_channels=[3,16, 24, 32, 88, 720]

        self.matting = Matting(self.encoder.out_channels)

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(640, 98 * 2 + 3 + 4, bias=True)

        self.hm = nn.Conv2d(in_channels=32, out_channels=98, kernel_size=3, stride=1, padding=1, bias=True)

        weight_init(self.fc)
        weight_init(self.hm)
    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        bs=x.size(0)
        features=self.encoder(x)

        features=[x]+features
        [encx2, encx4, encx8, encx16] = self.matting(features)


        fmx16 = self._avg_pooling(encx16)
        fmx8 = self._avg_pooling(encx8)
        fmx4 = self._avg_pooling(encx4)
        fmx2 = self._avg_pooling(encx2)
        fm = torch.cat([fmx2,fmx4, fmx8, fmx16], dim=1)

        fm = fm.view(bs, -1)
        x = self.fc(fm)

        hm = self.hm(encx2)

        return x,hm, [encx2, encx4, encx8, encx16, x]



class COTRAIN(nn.Module):
    def __init__(self,inference=False):
        super(COTRAIN, self).__init__()

        self.inference=inference
        self.student=Net()
        self.teacher=TeacherNet()

        self.MSELoss=nn.MSELoss()


        self.BCELoss     = nn.BCEWithLogitsLoss(reduction='none')


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


        losses=losses*weights
        loss = torch.sum(torch.mean(losses , dim=[0]))

        return loss


    def loss(self,predict_keypoints, label_keypoints):

        landmark_label = label_keypoints[:, :98*2]
        pose_label = label_keypoints[:, 196:199]

        cls_label=label_keypoints[:,199:199+4]
        # leye_cls_label = label_keypoints[:, 199]
        # reye_cls_label = label_keypoints[:, 200]
        # mouth_cls_label = label_keypoints[:, 201]
        # big_mouth_cls_label = label_keypoints[:, 202]

        landmark_weights=label_keypoints[:,199+4:199+4+196]

        cls_weights = label_keypoints[:, -4:]



        landmark_predict = predict_keypoints[:, :98*2]
        pose_predict = predict_keypoints[:, 196:199]
        # leye_cls_predict = predict_keypoints[:, 199]
        # reye_cls_predict = predict_keypoints[:, 200]
        # mouth_cls_predict = predict_keypoints[:, 201]
        # big_mouth_cls_predict = predict_keypoints[:, 202]
        cls_label_predict= predict_keypoints[:, 199:199+4]


        loss = self._wing_loss(landmark_predict, landmark_label,weights=landmark_weights)

        loss_pose = self.MSELoss(pose_predict, pose_label)

        cls_loss=self.BCELoss  ( cls_label_predict,cls_label)
        cls_loss=cls_loss*cls_weights




        cls_loss=torch.sum(cls_loss)/torch.sum(cls_weights)

        # leye_loss =  self.BCELoss  (leye_cls_predict, leye_cls_label)
        # reye_loss =  self.BCELoss  (reye_cls_predict, reye_cls_label)
        #
        # mouth_loss =  self.BCELoss  (mouth_cls_predict, mouth_cls_label)
        # mouth_loss_big =  self.BCELoss  (big_mouth_cls_predict, big_mouth_cls_label)
        # mouth_loss = mouth_loss + mouth_loss_big


        return loss + loss_pose + cls_loss

    def hm_loss(self,predict_hm, label_hm):

        bs=label_hm.size(0)
        
        hm_loss =  self.BCELoss(predict_hm, label_hm)

        hm_loss=torch.sum(hm_loss)/bs/64./64.
        return hm_loss





    def postp(self,hm):
        bs=hm.size(0)
        print(hm.size())
        hm=hm.reshape([bs,98,-1])

        hm=torch.argmax(hm,dim=2)


        X=hm%64
        Y=hm//64

        loc=torch.stack([X,Y],dim=2).float()/64


        return loc
    def forward(self, x,gt=None,gt_hm=None):

        student_pre,student_hm,student_fms=self.student(x)

        teacher_pre,teacher_hm, teacher_fms = self.teacher(x)



        if self.inference:
            # teacher_pre[:,-4:]=torch.nn.Sigmoid()(teacher_pre[:,-4:])
            # teacher_hm = torch.nn.Sigmoid()(teacher_hm)
            #
            # loc=self.postp(teacher_hm)

            return student_pre#,teacher_hm

        distill_loss=self.distill_loss(student_fms,teacher_fms)

        student_loss=self.loss(student_pre,gt)

        student_hm_loss=self.hm_loss(student_hm,gt_hm)

        student_loss=student_loss+student_hm_loss
        teacher_loss=self.loss(teacher_pre,gt)

        teacher_hm_loss = self.hm_loss(teacher_hm, gt_hm)
        teacher_loss=teacher_loss+teacher_hm_loss

        return student_loss,teacher_loss,distill_loss,student_pre,teacher_pre








if __name__=='__main__':
    import torch
    import torchvision

    from thop import profile

    # dummy_x = torch.randn(1, 3, 288, 160, device='cpu')

    model = COTRAIN(inference=True)

    input = torch.randn(1, 3, 128, 128)
    flops, params = profile(model, inputs=(input,))
    print(flops/1024/1024)

