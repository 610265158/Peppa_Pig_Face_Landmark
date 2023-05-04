##import pandas as pd
import sys
sys.path.append('.')

from lib.core.base_trainer.model import COTRAIN

import torch
import time
import argparse
import numpy as np
from tqdm import tqdm
import os
from  train_config import  config as cfg
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
def load_test_f(data_dir):
    txt_list=os.listdir(data_dir)
    txt_list=[x for x in txt_list if 'txt' in x]
    df={}
    for txt in txt_list:
        cls=txt.rsplit('.')[0].rsplit('_')[-1]

        with open(os.path.join(data_dir,txt)) as f:
            data_list=f.readlines()
        df[cls]=data_list
        # for line in data_list:
        #     line = line.split()

            # one_item={'fn':line[-1],
            #           'gt':np.array(line[:98*2],dtype=np.float32).reshape([-1,2])}

    return df


def augmentationCropImage( img, bbox, joints=None, is_training=True):
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

    return img, joints

def nme(target, preds):
    target = np.reshape(target, [-1, 98, 2])
    preds = np.reshape(preds, [-1, 98, 2])

    norm = np.linalg.norm(target[:, 60, :] - target[:, 72, :], axis=-1)

    distance = np.mean(np.linalg.norm(preds - target, axis=-1), axis=-1) / norm

    nme = np.mean(distance)

    return nme

def do_eval(data_dir,model,input_size):



    WFLW_df=load_test_f(os.path.join(data_dir,'WFLW_annotations/list_98pt_test'))

    model.to(device)
    for k,v in WFLW_df.items():
        data_list=v

        nme_list=[]
        for dp in tqdm(data_list):
            dp = dp.split()
            kps = dp[:98 * 2]
            fn = dp[-1]

            image=cv2.imread(os.path.join(data_dir,'WFLW_images',fn))
            kps = np.array(kps, dtype=np.float32).reshape([-1, 2])

            bbox = [float(np.min(kps[:, 0])), float(np.min(kps[:, 1])), float(np.max(kps[:, 0])),
                    float(np.max(kps[:, 1]))]

            bbox = np.array(bbox)

            ### random crop and resize
            crop_image, label = augmentationCropImage(image, bbox, kps, False)

            h,w,c=crop_image.shape

            label[:,0]/=w
            label[:, 1] /= h

            crop_image=cv2.resize(crop_image,(input_size,input_size))

            crop_image = np.transpose(crop_image, axes=[2, 0, 1])
            crop_image=crop_image.astype(np.float32)
            crop_image /= 255.
            crop_image=np.expand_dims(crop_image,axis=0)
            crop_image=torch.from_numpy(crop_image).to(device)
            preds=model(crop_image)[0]
            preds=preds.cpu().detach().numpy()
            preds =preds[:98*2]

            nme_score=nme(kps,preds)

            nme_list.append(nme_score)
        print('for cls:',k, ' nme:',np.mean(nme_list))


def get_model(weight,model='teacher'):

    model=COTRAIN(inference=model)
    model.eval()
    state_dict = torch.load(weight, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    return model

def main(data_dir,weight,input_size):

    model=get_model(weight)

    do_eval(data_dir,model,input_size)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Start train.')

    parser.add_argument('--weight', dest='weight', type=str, default=None, \
                        help='the weight to use')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default=None, \
                        help='the data_dir to use')
    parser.add_argument('--img_size', dest='img_size', type=int, default=256, \
                        help='the inputsize to use')
    parser.add_argument('--model', dest='model', type=str, default='teacher', \
                        help='teache or student')
    args = parser.parse_args()

    data_dir=args.data_dir
    weight=args.weight
    img_size=args.img_size
    model=args.model
    main(data_dir,weight,img_size,model)




