

import os
import numpy as np
from easydict import EasyDict as edict

config = edict()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config.TRAIN = edict()
#### below are params for dataiter
config.TRAIN.process_num = 4
config.TRAIN.prefetch_size = 30
############

config.TRAIN.num_gpu = 1
config.TRAIN.batch_size = 128
config.TRAIN.validatiojn_batch_size = 128
config.TRAIN.accumulation_batch_size=128
config.TRAIN.log_interval = 10                  ##10 iters for a log msg
config.TRAIN.epoch = 50
config.TRAIN.early_stop=20
config.TRAIN.test_interval=1

config.TRAIN.init_lr = 1.e-3
config.TRAIN.warmup_step=1500
config.TRAIN.weight_decay_factor = 5.e-4                                    ####l2
config.TRAIN.vis=False                                                      #### if to check the training data
config.TRAIN.mix_precision=False                                            ##use mix precision to speedup, tf1.14 at least
config.TRAIN.opt='AdamW'                                                     ##Adam or SGD
config.TRAIN.gradient_clip=5

config.MODEL = edict()
config.MODEL.model_path = './models/'                                        ## save directory
config.MODEL.hin = 128                                                      # input size during training , 128,160,   depends on
config.MODEL.win = 128

config.MODEL.out_channel=136+3+4    # output vector    68 points , 3 headpose ,4 cls params,(left eye, right eye, mouth, big mouth open)

config.MODEL.pretrained_model='./models/fold0_epoch_8_val_loss_12.163321_val_mse_0.249408.pth'
config.DATA = edict()

config.DATA.root_path=''
config.DATA.train_f_path='train.csv'
config.DATA.val_f_path='val.csv'

############the model is trained with RGB mode
config.DATA.root_path='../tmp_crop_data_face_landmark_pytorch'

config.DATA.base_extend_range=[0.1,0.2]                 ###extand
config.DATA.scale_factor=[0.7,1.35]                     ###scales

config.DATA.symmetry = [(0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9), (8, 8),
            (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
            (31, 35), (32, 34),
            (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
            (48, 54), (49, 53), (50, 52), (55, 59), (56, 58), (60, 64), (61, 63), (65, 67)]





weights=[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,                                    #####bouding
         1.5,1.5,1.5,1.5,1.5,       1.5,1.5,1.5,1.5,1.5,                                          #####eyebows
         1.,1.,1.,1.,1.,1.,1.,1.,1.,                                                              #####nose
         1.5,1.5,1.5,1.5,1.5,1.5,    1.5,1.5,1.5,1.5,1.5,1.5,                                     ####eyes
         1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.                              #####mouth
       ]
weights_xy=[[x,x] for x in weights]

config.DATA.weights = np.array(weights_xy,dtype=np.float32).reshape([-1])






