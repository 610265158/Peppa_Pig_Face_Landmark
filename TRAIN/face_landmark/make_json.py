import os
import random
import numpy as np
import json
import traceback
import cv2
import pandas as pd

from tqdm import tqdm
'''
i decide to merge more data from CelebA, the data anns will be complex, so json maybe a better way. 
'''





data_dir='/media/lz/ssd_2/coco_data/facelandmark/PUB'      ########points to your director,300w
#celeba_data_dir='CELEBA'                      ########points to your director,CELEBA


train_json='./train.csv'
val_json='./val.csv'
save_dir='../tmp_crop_data_face_landmark_pytorch'

if not os.access(save_dir,os.F_OK):
    os.mkdir(save_dir)

def GetFileList(dir, fileList):
    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):

            # if s == "pts":
            #     continue
            newDir=os.path.join(dir,s)
            GetFileList(newDir, fileList)
    return fileList




pic_list=[]
GetFileList(data_dir,pic_list)

pic_list=[x for x in pic_list if '.jpg' in x or 'png' in x or 'jpeg' in x  ]


ratio=0.95
train_list=[x for x in pic_list if 'AFW' not in x]
val_list=[x for x in pic_list if 'AFW'  in x]

# train_list=[x for x in pic_list if '300W/' not in x]
# val_list=[x for x in pic_list if '300W/' in x]


def process_data(data_list,csv_nm):


    global cnt
    image_list=[]
    keypoint_list=[]

    for pic in tqdm(data_list):
        one_image_ann={}

        ### image_path
        one_image_ann['image_path_raw']=pic

        #### keypoints
        pts=pic.rsplit('.',1)[0]+'.pts'
        if os.access(pic,os.F_OK) and  os.access(pts,os.F_OK):
            try:
                tmp=[]
                with open(pts) as p_f:
                    labels=p_f.readlines()[3:-1]
                for _one_p in labels:
                    xy = _one_p.rstrip().split(' ')
                    tmp.append([float(xy[0]),float(xy[1])])

                one_image_ann['keypoints'] = tmp

                label = np.array(tmp).reshape((-1, 2))
                bbox = [float(np.min(label[:, 0])), float(np.min(label[:, 1])), float(np.max(label[:, 0])), float(np.max(label[:, 1]))]
                one_image_ann['bbox'] = bbox

                ### placeholder
                one_image_ann['attr'] = None

                ###### crop it

                image=cv2.imread(one_image_ann['image_path_raw'],cv2.IMREAD_COLOR)

                h,w,c=image.shape

                ##expanded for
                bbox_int = [int(x) for x in bbox]
                bbox_width = bbox_int[2] - bbox_int[0]
                bbox_height = bbox_int[3] - bbox_int[1]

                center_x=(bbox_int[2] + bbox_int[0])//2
                center_y=(bbox_int[3] + bbox_int[1])//2

                x1=int(center_x-bbox_width*2)
                x1=x1 if x1>=0 else 0

                y1 = int(center_y - bbox_height*2)
                y1 = y1 if y1 >= 0 else 0

                x2 = int(center_x + bbox_width*2)
                x2 = x2 if x2 <w else w

                y2 = int(center_y + bbox_height*2)
                y2 = y2 if y2 <h else h

                crop_face=image[y1:y2,x1:x2,...]


                hh,ww,cc=crop_face.shape

                if max(hh,ww)>512:
                    scale=512/max(hh,ww)
                else:
                    scale=1
                crop_face=cv2.resize(crop_face,None,fx=scale,fy=scale)

                one_image_ann['bbox'][0] *= scale
                one_image_ann['bbox'][1] *= scale
                one_image_ann['bbox'][2] *= scale
                one_image_ann['bbox'][3] *= scale


                x1*=scale
                y1 *= scale
                x2 *= scale
                y2 *= scale
                for i in range(len(one_image_ann['keypoints'])):
                    one_image_ann['keypoints'][i][0]*= scale
                    one_image_ann['keypoints'][i][1]*= scale

                fname= one_image_ann['image_path_raw'].split('PUB/')[-1]

                fname=fname.replace('/','_').replace('/','_')


                # cv2.imwrite(one_image_ann['image_name'],crop_face)


                one_image_ann['bbox'][0] -= x1
                one_image_ann['bbox'][1] -= y1
                one_image_ann['bbox'][2] -= x1
                one_image_ann['bbox'][3] -= y1

                for i in range(len(one_image_ann['keypoints'])):
                    one_image_ann['keypoints'][i][0]-=x1
                    one_image_ann['keypoints'][i][1]-=y1


                keypoint=list(np.array(one_image_ann['keypoints']).reshape(-1).astype(np.float32))
                # [x1,y1,x2,y2]=[int(x) for x in one_image_ann['bbox']]
                #
                # cv2.rectangle(crop_face,(x1,y1),(x2,y2),thickness=2,color=(255,0,0))
                #
                # landmark=np.array(one_image_ann['keypoints'])
                #
                # for _index in range(landmark.shape[0]):
                #     x_y = landmark[_index]
                #     # print(x_y)
                #     cv2.circle(crop_face, center=(int(x_y[0] ),
                #                                  int(x_y[1] )),
                #                color=(255, 0, 0), radius=2, thickness=4)
                #
                #
                # cv2.imshow('ss', crop_face)
                # cv2.waitKey(0)

                image_list.append(fname)
                keypoint_list.append(keypoint)
                # json_list.append(one_image_ann)
            except:
                print(pic)

                print(traceback.print_exc())


    # with open(json_nm, 'w') as f:
    #     json.dump(json_list, f, indent=2)

    data_dict={'image':image_list,
               'keypoint':keypoint_list}
    df=pd.DataFrame(data_dict)

    df.to_csv(csv_nm,index=False)


process_data(train_list,train_json)


process_data(val_list,val_json)










