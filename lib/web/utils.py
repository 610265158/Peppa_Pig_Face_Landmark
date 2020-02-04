import cv2
import numpy as np
import json


def parse_img(img_str):

    image_data = np.fromstring(img_str, np.uint8)
    image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)


    # cv2.imshow('parse',image_data)
    # cv2.waitKey(0)
    return image_data





def parse_as_dict(landmarks):



    res=[]
    for face_index in range(landmarks.shape[0]):

        cur_face={}

        bbox = [int(np.min(landmarks[face_index][:, 0])),
                int(np.min(landmarks[face_index][:, 1])),
                int(np.max(landmarks[face_index][:, 0])),
                int(np.max(landmarks[face_index][:, 1]))]
        cur_face["bbox"]=bbox


        landmarks_list=[]

        for landmarks_index in range(landmarks[face_index].shape[0]):
            x_y = landmarks[face_index][landmarks_index]
            x_y_float=[int(x_y[0]),int(x_y[1])]
            landmarks_list.append(x_y_float)

        cur_face["landmark"] = landmarks_list



        res.append(cur_face)


    return res