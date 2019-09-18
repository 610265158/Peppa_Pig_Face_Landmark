import cv2
import numpy as np
import time

from lib.core.api.face_keypoint import Keypoints
from lib.core.api.face_detector import FaceDetector
from config import config as cfg

class FaceAna():

    '''
    by default the top3 facea sorted by area will be calculated for time reason
    '''
    def __init__(self):
        self.face_detector = FaceDetector()
        self.face_landmark =Keypoints()
        self.top_k=cfg.DETECT.topk

        ###another thread should do detector in a slow way and update the track_box
        self.track_box=None
        self.previous_image=None
        self.previous_box=None
        self.diff_thres=5
        self.iou_thres=cfg.TRACE.iou_thres

    def run(self,image):

        start = time.time()
        if self.diff_frames(self.previous_image,image):
            boxes = self.face_detector(image)
            self.previous_image=image
            #self.track_box=None
        else:
            boxes=self.track_box
            self.previous_image = image
        print('facebox detect cost',time.time()-start)

        if boxes.shape[0]>self.top_k:
            boxes=self.sort(boxes)

        boxes=self.judge_boxs(self.track_box,boxes)

        boxes_return = np.array(boxes)

        landmarks,states=self.face_landmark.run(image,boxes)

        if 1:
            track=[]
            for i in range(landmarks.shape[0]):
                track.append([np.min(landmarks[i][:,0]),np.min(landmarks[i][:,1]),np.max(landmarks[i][:,0]),np.max(landmarks[i][:,1])])
            tmp_box=np.array(track)
            self.track_box = self.judge_boxs(boxes_return, tmp_box)
        # else:
        #     self.track_box = self.judge_boxs(boxes_return,)

        return boxes_return,landmarks,states

    def diff_frames(self,previous_frame,image):
        if previous_frame is None:
            return True
        else:

            previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

            image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            _diff = cv2.absdiff(previous_frame, image)

            diff=np.sum(_diff)/previous_frame.shape[0]/previous_frame.shape[1]
            #print(diff)
            if diff>self.diff_thres:
                return True
            else:
                return False
    def sort(self,bboxes):
        if self.top_k >100:
            return bboxes
        area=[]
        for bbox in bboxes:

            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            area.append(bbox_height*bbox_width)
        area=np.array(area)

        picked=area.argsort()[-self.top_k:][::-1]
        sorted_bboxes=[bboxes[x] for x in picked]
        return np.array(sorted_bboxes)

    def judge_boxs(self,previuous_bboxs,now_bboxs):
        def iou(rec1, rec2):

            # computing area of each rectangles
            S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
            S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

            # computing the sum_area
            sum_area = S_rec1 + S_rec2

            # find the each edge of intersect rectangle
            x1 = max(rec1[0], rec2[0])
            y1 = max(rec1[1], rec2[1])
            x2 = min(rec1[2], rec2[2])
            y2 = min(rec1[3], rec2[3])

            # judge if there is an intersect
            intersect =max(0,x2-x1) * max(0,y2-y1)

            return intersect / (sum_area - intersect)


        if previuous_bboxs is None:
            return now_bboxs

        result=[]
        contain=False
        for i in range(now_bboxs.shape[0]):
            for j in range(previuous_bboxs.shape[0]):
                if iou(now_bboxs[i], previuous_bboxs[j]) > self.iou_thres:
                    result.append(previuous_bboxs[j])
                    contain=True
                    break
            if contain:
                contain = False

            else:
                result.append(now_bboxs[i])
                contain=False

        return np.array(result)






    def reset(self):
        self.previous_image = None
