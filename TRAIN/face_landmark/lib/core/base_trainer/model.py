import math
from functools import partial
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig


# from lib.core.base_trainer.mobileone import MobileOneBlock
class SeparableConv2d(nn.Module):
    """ Separable Conv
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding=0, bias=False,
                 channel_multiplier=1., pw_kernel_size=1):
        super(SeparableConv2d, self).__init__()

        self.conv_dw = nn.Sequential(nn.Conv2d(
            int(in_channels * channel_multiplier), int(in_channels * channel_multiplier), kernel_size,
            stride=stride, dilation=dilation, padding=padding, groups=int(in_channels * channel_multiplier)),
            nn.BatchNorm2d(in_channels)

        )
        self.conv_pw = nn.Conv2d(
            int(in_channels * channel_multiplier), out_channels, pw_kernel_size, padding=0, bias=bias)

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
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]

        x = self.pool(x)

        x = F.interpolate(x, size=size)
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=512):
        super(ASPP, self).__init__()

        rate1, rate2, rate3 = tuple(atrous_rates)

        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=2, bias=False, dilation=rate1)

        self.conv3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=4, bias=False, dilation=rate2)

        self.bn_act = nn.Sequential(nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True))

        self.fm_pool = ASPPPooling(in_channels=in_channels, out_channels=out_channels // 4)

        self.project = nn.Sequential(
            nn.Conv2d(out_channels // 4 * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        fm1 = self.conv1(x)
        fm2 = self.conv2(x)
        fm4 = self.conv3(x)

        fm_pool = self.fm_pool(x)

        res = torch.cat([fm1, fm2, fm4, fm_pool], dim=1)

        res = self.bn_act(res)

        return self.project(res)


class FeatureFuc(nn.Module):
    def __init__(self, inchannels=128):
        super(FeatureFuc, self).__init__()

        self.block1 = InvertedResidual(cnf=InvertedResidualConfig(inchannels, 5, 256, inchannels, False, "RE", 1, 1, 1),
                                       norm_layer=partial(nn.BatchNorm2d, ))

        self.block2 = InvertedResidual(cnf=InvertedResidualConfig(inchannels, 5, 256, inchannels, False, "RE", 1, 1, 1),
                                       norm_layer=partial(nn.BatchNorm2d, ))

    def forward(self, x):
        y1 = self.block1(x)

        y2 = self.block2(y1)

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
        return x * self.cSE(x) + x * self.sSE(x)


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
                in_channels + skip_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))

        # self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        if use_second_conv:
            self.conv2 = nn.Sequential(nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
        else:
            self.conv2 = nn.Identity()

        if use_attention:
            self.attention2 = SCSEModule(in_channels=out_channels)
        else:
            self.attention2 = nn.Identity()

    def forward(self, x, skip=None):

        # x= upsample_x_like_y(x,skip)
        x = F.interpolate(x, scale_factor=2,mode='bilinear')
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


class Decoder(nn.Module):

    def __init__(self, encoder_channels):
        super(Decoder, self).__init__()

        self.aspp = ASPP(encoder_channels[-1], [2, 4, 8],out_channels=256)

        # self.upsampler1 = DecoderBlock(256, encoder_channels[-2], 256, \
        #                                use_separable_conv=True, \
        #                                use_attention=True,
        #                                kernel_size=3)

        self.upsampler2 = DecoderBlock(256, encoder_channels[-2], 128, \
                                       use_separable_conv=True, \
                                       use_attention=False,
                                       use_second_conv=True,
                                       kernel_size=3)

        self.apply(weight_init)

    def forward(self, features):
        img, encx2, encx4, encx8, encx16 = features

        ## add extra feature

        encx16 = self.aspp(encx16)

        # decx8 = self.upsampler1(encx16, encx8)
        decx8 = self.upsampler2(encx16, encx8)

        #### semantic predict

        return [ decx8, encx16]


class Net(nn.Module):
    def __init__(self, inp_size=(128, 128)):
        super(Net, self).__init__()

        self.input_size = inp_size
        self.encoder = timm.create_model(model_name='mobilenetv3_large_100.ra_in1k',
                                         pretrained=True,
                                         features_only=True,
                                         out_indices=[0, 1, 2, 4],
                                         in_chans=3,
                                         output_stride=16,
                                         )

        # self.encoder.blocks[4][1]=nn.Identity()

        self.encoder.blocks[6] = nn.Identity()

        self.encoder_out_channels = [3, 16, 24, 40, 160]  # mobilenetv3

        self.decoder = Decoder(self.encoder_out_channels)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(384,  3 + 4, bias=True)

        self.hm = nn.Conv2d(in_channels=128, out_channels=98*3, kernel_size=1, stride=1, padding=0, bias=True)

        weight_init(self.fc)
        weight_init(self.hm)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        bs = x.size(0)

        features = self.encoder(x)

        features = [x] + features

        [ decx8, decx16] = self.decoder(features)

        fmx16 = self._avg_pooling(decx16)
        fmx8 = self._avg_pooling(decx8)


        fm = torch.cat([ fmx8, fmx16], dim=1)

        fm = fm.view(bs, -1)
        x = self.fc(fm)

        hm = self.hm(decx8)


        return x, hm, [hm]



class TeacherNet(nn.Module):
    def __init__(self, inp_size=(128, 128)):
        super(TeacherNet, self).__init__()
        self.input_size = inp_size
        self.encoder = timm.create_model(model_name='hrnet_w18',
                                         pretrained=True,
                                         features_only=True,
                                         out_indices=[0, 1, 2, 3],
                                         in_chans=3,
                                         )

        self.encoder.out_channels = [3, 64, 128, 256, 512]

        self.decoder = Decoder(self.encoder.out_channels)

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(384,  3 + 4, bias=True)

        self.hm = nn.Conv2d(in_channels=128, out_channels=98*3, kernel_size=1, stride=1, padding=0, bias=True)

        weight_init(self.fc)
        weight_init(self.hm)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        bs = x.size(0)
        features = self.encoder(x)

        features = [x] + features
        [ decx8, decx16] = self.decoder(features)

        fmx16 = self._avg_pooling(decx16)
        fmx8 = self._avg_pooling(decx8)


        fm = torch.cat([ fmx8, fmx16], dim=1)

        fm = fm.view(bs, -1)
        x = self.fc(fm)

        hm = self.hm(decx8)

        return x, hm, [hm]



class AWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1, use_weight_map=True):
        super(AWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        self.use_weight_map = use_weight_map

    def __repr__(self):
        return "AWingLoss()"

    def generate_weight_map(self, heatmap, k_size=3, w=10):
        dilate = F.max_pool2d(heatmap, kernel_size=k_size, stride=1, padding=1)
        weight_map = torch.where(dilate < 0.2, torch.zeros_like(heatmap), torch.ones_like(heatmap))
        return w * weight_map + 1

    def forward(self, output, groundtruth):
        """
        input:  b x n x h x w
        output: b x n x h x w => 1
        """
        delta = (output - groundtruth).abs()
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - groundtruth))) * (
                    self.alpha - groundtruth) * \
            (torch.pow(self.theta / self.epsilon, self.alpha - groundtruth - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * \
            torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - groundtruth))
        loss = torch.where(delta < self.theta,
                           self.omega * torch.log(1 + torch.pow(delta / self.epsilon, self.alpha - groundtruth)),
                           (A * delta - C))
        if self.use_weight_map:
            weight = self.generate_weight_map(groundtruth)
            loss = loss * weight
        return loss


class COTRAIN(nn.Module):
    def __init__(self, inference=None, inp_size=(128, 128)):
        super(COTRAIN, self).__init__()

        self.inference = inference
        self.student = Net(inp_size)
        self.teacher = TeacherNet(inp_size)

        self.MSELoss = nn.MSELoss()
        self.MSELoss_no_reduction = nn.MSELoss(reduction='none')
        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')



        self.Awing = AWingLoss()
        if inference=='teacher':
            self.run_with_teacher=True

    def distill_loss(self, student_pres, teacher_pres):

        num_level = len(student_pres)
        loss = 0
        for i in range(num_level):
            loss += self.MSELoss(student_pres[i], teacher_pres[i].detach())

        return loss / num_level

    def criterion(self, y_pred, y_true):

        return 0.5 * self.BCELoss(y_pred, y_true) + 0.5 * self.DiceLoss(y_pred, y_true)

    def _wing_loss(self, landmarks, labels, w=10.0, epsilon=2.0, weights=1.):
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

        

        return losses

    def loss(self, predict_keypoints, label_keypoints):

        pose_label = label_keypoints[:, 98*2:98*2+3]

        cls_label = label_keypoints[:, 98*2+3:98*2+3 + 4]
        # leye_cls_label = label_keypoints[:, 199]
        # reye_cls_label = label_keypoints[:, 200]
        # mouth_cls_label = label_keypoints[:, 201]
        # big_mouth_cls_label = label_keypoints[:, 202]

        cls_weights = label_keypoints[:, -4:]

        pose_predict = predict_keypoints[:, :3]
        # leye_cls_predict = predict_keypoints[:, 199]
        # reye_cls_predict = predict_keypoints[:, 200]
        # mouth_cls_predict = predict_keypoints[:, 201]
        # big_mouth_cls_predict = predict_keypoints[:, 202]
        cls_label_predict = predict_keypoints[:, 3:3 + 4]

        #loss = self._wing_loss(landmark_predict, landmark_label, weights=landmark_weights)

        loss_pose = self.MSELoss(pose_predict, pose_label)

        cls_loss = self.BCELoss(cls_label_predict, cls_label)
        cls_loss = cls_loss * cls_weights

        cls_loss = torch.sum(cls_loss) / torch.sum(cls_weights)

        # leye_loss =  self.BCELoss  (leye_cls_predict, leye_cls_label)
        # reye_loss =  self.BCELoss  (reye_cls_predict, reye_cls_label)
        #
        # mouth_loss =  self.BCELoss  (mouth_cls_predict, mouth_cls_label)
        # mouth_loss_big =  self.BCELoss  (big_mouth_cls_predict, big_mouth_cls_label)
        # mouth_loss = mouth_loss + mouth_loss_big

        return   loss_pose + cls_loss


    def offside_loss(self,pre,gt,weight):

        loss=self._wing_loss(pre,gt)


        loss=loss*weight

        loss=torch.sum(loss)/torch.sum(weight)

        return loss
    def hm_loss(self, predict_hm, label_hm):

        bs = label_hm.size(0)

        hm=label_hm[:,:98,...]
        hm_prd = predict_hm[:, :98, ...]
        hm_loss = self.Awing(hm_prd, hm)

        hm_loss = torch.mean(hm_loss)

        offside_pre_x=predict_hm[:,98:2*98,...]
        offside_gt_x = label_hm[:, 98:2*98, ...]

        offside_loss_x=self.offside_loss(offside_pre_x,offside_gt_x,label_hm[:,:98,...])

        offside_pre_y = predict_hm[:, 2 * 98:, ...]
        offside_gt_y = label_hm[:, 2 * 98: , ...]

        offside_loss_y = self.offside_loss(offside_pre_y, offside_gt_y, label_hm[:, :98, ...])

        ##offside_loss

        return hm_loss+offside_loss_x+offside_loss_y

    def postp(self, hm):

        #
        hm_score=hm[:,:98,...]

        hm_H = hm.size(2)
        hm_W = hm.size(3)
        bs = hm.size(0)

        hm_score = hm_score.reshape([bs, 98, -1])

        score,hm_indx = torch.max(hm_score, dim=2)

        #### add offside

        offside_x=hm[:,98:2*98,...].reshape([bs, 98, -1])
        offside_y = hm[:, 2*98:, ...].reshape([bs, 98, -1])


        gether_indx=hm_indx.unsqueeze(-1)
        offside_x = torch.gather(offside_x,dim=-1,index=gether_indx).squeeze(-1)
        offside_y = torch.gather(offside_y,dim=-1,index=gether_indx).squeeze(-1)


        X = hm_indx % hm_W
        Y = hm_indx // hm_W


        X_fix = X + offside_x
        Y_fix = Y + offside_y

        loc = torch.stack([X, Y], dim=2).float()
        loc[..., 0] /= hm_W
        loc[..., 1] /= hm_H
        loc = loc.view(bs, -1)



        loc_fix = torch.stack([X_fix, Y_fix], dim=2).float()
        loc_fix[..., 0] /= hm_W
        loc_fix[..., 1] /= hm_H
        loc_fix = loc_fix.view(bs, -1)

        return loc,loc_fix,score

    def forward(self, x, gt=None, gt_hm=None):

        student_pre, student_hm, student_fms = self.student(x)

        teacher_pre, teacher_hm, teacher_fms = self.teacher(x)

        if self.inference :
            if self.inference=='teacher':
                hm_used=teacher_hm
            else:
                hm_used=student_hm
            teacher_pre, teacher_pre_full,score = self.postp(hm_used)
            return teacher_pre_full  ,score

        distill_loss = self.distill_loss(student_fms, teacher_fms)

        student_loss = self.loss(student_pre, gt)

        student_hm_loss = self.hm_loss(student_hm, gt_hm)

        student_loss = student_loss + student_hm_loss
        teacher_loss = self.loss(teacher_pre, gt)

        teacher_hm_loss = self.hm_loss(teacher_hm, gt_hm)

        teacher_loss = teacher_loss + teacher_hm_loss

        ### decode hm
        student_pre,student_pre_full,_ = self.postp(student_hm)
        teacher_pre,teacher_pre_full,_ = self.postp(teacher_hm)

        return student_loss, teacher_loss, distill_loss, student_pre, student_pre_full, teacher_pre, teacher_pre_full


if __name__ == '__main__':
    import torch
    import torchvision

    from thop import profile
    from thop import clever_format

    model = COTRAIN(inference='teacher')

    input = torch.randn(1, 3, 256, 256)
    macs, params = profile(model, inputs=(input,))

    macs, params = clever_format([macs, params], "%.3f")

    print(macs)
    print(params)

