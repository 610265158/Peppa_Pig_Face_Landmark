import json

from lib.core.base_trainer.net_work import Train
from lib.dataset.dataietr import AlaskaDataIter
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd



from train_config import config as cfg

import setproctitle


setproctitle.setproctitle("pks")





def get_fold(df,n_folds):

    skf = KFold(n_splits=n_folds, shuffle=True, random_state=cfg.SEED)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df)):
        df.loc[val_idx, 'fold'] = fold

    return df


def setppm100asval(df):
    def func(fn):
        if 'PPM' in fn:
            return 0
        else:
            return 1
    df['fold']=df['image'].apply(func)


    return df


def main():





    train_df = pd.read_csv(cfg.DATA.train_f_path)

    val_df =pd.read_csv(cfg.DATA.val_f_path)


    ###build trainer
    trainer = Train(train_df=train_df,val_df=val_df,fold=0)

    ### train
    trainer.custom_loop()

if __name__=='__main__':
    main()

