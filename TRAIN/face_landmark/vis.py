import pandas as pd

from lib.dataset.dataietr import AlaskaDataIter
from train_config import config
from lib.core.base_trainer.model import COTRAIN

import torch
import time
import argparse

from torch.utils.data import DataLoader
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
from train_config import config as cfg
cfg.TRAIN.batch_size=1


df=pd.read_csv(cfg.DATA.val_f_path)
val_genererator = AlaskaDataIter(df,
                    img_root=cfg.DATA.root_path,
                    training_flag=False, shuffle=False)
val_ds=DataLoader(val_genererator,
           cfg.TRAIN.batch_size,
                  num_workers=1,shuffle=False)


def vis(weight):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model=COTRAIN(inference=True)
    model.eval()
    state_dict = torch.load(weight, map_location=device)
    model.load_state_dict(state_dict, strict=False)



    for step, (ids, images, kps) in enumerate(val_ds):


        # kps = kps.to(device).float()

        img_show = np.array(images)*255
        print(img_show.shape)

        img_show=np.transpose(img_show[0],axes=[1,2,0]).astype(np.uint8)
        img_show=np.ascontiguousarray(img_show)
        images=images.to(device)
        print(images.size())

        start=time.time()
        with torch.no_grad():
            res=model(images)
        res=res.detach().numpy()
        print(res)
        print('xxxx',time.time()-start)
        #print(res)




        landmark = np.array(res[0][0:136]).reshape([-1, 2])

        for _index in range(landmark.shape[0]):
            x_y = landmark[_index]
            #print(x_y)
            cv2.circle(img_show, center=(int(x_y[0] * 128),
                                         int(x_y[1] * 128)),
                       color=(255, 122, 122), radius=1, thickness=2)

        cv2.imshow('tmp',img_show)
        cv2.waitKey(0)


def load_checkpoint(net, checkpoint):
    # from collections import OrderedDict
    #
    # temp = OrderedDict()
    # if 'state_dict' in checkpoint:
    #     checkpoint = dict(checkpoint['state_dict'])
    # for k in checkpoint:
    #     k2 = 'module.'+k if not k.startswith('module.') else k
    #     temp[k2] = checkpoint[k]

    net.load_state_dict(torch.load(checkpoint,map_location=torch.device('cpu')), strict=True)
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Start train.')

    parser.add_argument('--model', dest='model', type=str, default=None, \
                        help='the model to use')

    args = parser.parse_args()


    vis(args.model)




