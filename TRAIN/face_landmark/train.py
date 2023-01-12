import json

from lib.core.base_trainer.net_work import Train
from lib.dataset.dataietr import AlaskaDataIter
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd



from train_config import config as cfg

import setproctitle


setproctitle.setproctitle("kps")







def main():
    extra=cfg.DATA.extra_data
    with open(cfg.DATA.train_f_path,mode='r') as f:
        train_df=f.readlines()

    if extra:
        with open('extradata.txt', mode='r') as f:
            extra_df = f.readlines()
            train_df+=extra_df
            # train_df=extra_df
            # print(train_df[-1])
    with open(cfg.DATA.val_f_path,mode='r') as f:
        val_df=f.readlines()



    ###build trainer
    trainer = Train(train_df=train_df,val_df=val_df,fold=0)

    ### train
    trainer.custom_loop()

if __name__=='__main__':
    main()
