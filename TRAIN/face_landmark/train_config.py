

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
config.TRAIN.batch_size = 64
config.TRAIN.validatiojn_batch_size = 64
config.TRAIN.accumulation_batch_size=64
config.TRAIN.log_interval = 10                  ##10 iters for a log msg
config.TRAIN.epoch = 100
config.TRAIN.early_stop=20
config.TRAIN.test_interval=1

config.TRAIN.init_lr = 2.e-3
config.TRAIN.warmup_step=1500
config.TRAIN.weight_decay_factor = 5.e-4                                    ####l2
config.TRAIN.vis=False                                                      #### if to check the training data
config.TRAIN.mix_precision=True                                            ##use mix precision to speedup, tf1.14 at least
config.TRAIN.opt='Adamw'                                                     ##Adam or SGD
config.TRAIN.gradient_clip=-5

config.MODEL = edict()
config.MODEL.model_path = './models/'                                        ## save directory
config.MODEL.hin =  128                                                     # input size during training , 128,160,   depends on
config.MODEL.win = 128

config.MODEL.out_channel=98*2+3+4    # output vector    68 points , 3 headpose ,4 cls params,(left eye, right eye, mouth, big mouth open)

config.MODEL.pretrained_model=None
config.DATA = edict()

config.DATA.root_path='../WFLW_images'
config.DATA.train_f_path='../WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt'
config.DATA.val_f_path='../WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt'
config.DATA.extra_data=True


config.DATA.base_extend_range=[0.1,0.2]                 ###extand
config.DATA.scale_factor=[0.7,1.35]                     ###scales

# config.DATA.symmetry = [(0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9), (8, 8),
#             (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
#             (31, 35), (32, 34),
#             (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
#             (48, 54), (49, 53), (50, 52), (55, 59), (56, 58), (60, 64), (61, 63), (65, 67)]

config.DATA.symmetry = [(0, 32), (1, 31), (2, 30), (3, 29), (4, 28), (5, 27), (6, 26), (7, 25), (8, 24),
                        (9,23),(10,22),(11,21),(12,20),(13,19),(14,18),(15,17),(16,16),
                        ##
                        (33,46),(34,45),(35,44),(36,43),(37,42),(38,50),(39,49),(40,48),(41,47),
                        ##
                        (60,72),(61,71),(62,70),(63,69),(64,68),(65,75),(66,74),(67,73),(96,97),
                        ##
                        (51,51),(52,52),(53,53),(54,54),
                        ##
                        (55,59),(56,58),(57,57),
                        ##
                        (76,82),(77,81),(78,80),(79,79),
                        (87,83),(86,84),(85,85),
                        (88,92),(89,91),(90,90),
                        (95,93),(94,94)
            ]





config.SEED=42


from lib.utils.seed_util import seed_everything

seed_everything(config.SEED)






