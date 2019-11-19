import cv2
import numpy as np
import time

from lib.core.api.face_landmark import FaceLandmark
from lib.core.api.face_detector import FaceDetector
from lib.core.LK.lk import GroupTrack,OneEuroFilter,EmaFilter

from config import config as cfg

class FaceAna():

    '''
    by default the top3 facea sorted by area will be calculated for time reason
    '''
    def __init__(self):

        
        self.face_detector = FaceDetector()
        self.face_landmark = FaceLandmark()
        self.trace = GroupTrack()



        ###another thread should run detector in a slow way and update the track_box
        self.track_box=None
        self.previous_image=None
        self.previous_box=None

        self.diff_thres=5
        self.top_k = cfg.DETECT.topk
        self.min_face=cfg.DETECT.min_face
        self.iou_thres=cfg.TRACE.iou_thres
        self.alpha=cfg.TRACE.smooth_box

        if 'ema' in cfg.TRACE.ema_or_one_euro:
            self.filter = EmaFilter(self.alpha)
        else:
            self.filter = OneEuroFilter()

    def run(self,image):

        ###### run detector
        if self.diff_frames(self.previous_image,image):
            boxes = self.face_detector(image)
            self.previous_image=image
            boxes = self.judge_boxs(self.track_box, boxes)

        else:
            boxes=self.track_box
            self.previous_image = image



        boxes=self.sort_and_filter(boxes)

        boxes_return = np.array(boxes)


        #### batch predict for face landmark
        landmarks,states=self.face_landmark.batch_call(image,boxes)


        #### calculate the headpose for the whole image
        landmarks = self.trace.calculate(image, landmarks)


        #### refine the bboxes
        track=[]
        for i in range(landmarks.shape[0]):
            track.append([np.min(landmarks[i][:,0]),np.min(landmarks[i][:,1]),np.max(landmarks[i][:,0]),np.max(landmarks[i][:,1])])
        tmp_box=np.array(track)

        self.track_box = self.judge_boxs(boxes_return, tmp_box)


        return self.track_box,landmarks,states

    def diff_frames(self,previous_frame,image):
        '''
        diff value for two value,
        determin if to excute the detection

        :param previous_frame:  RGB  array
        :param image:           RGB  array
        :return:                True or False
        '''
        if previous_frame is None:
            return True
        else:

            _diff = cv2.absdiff(previous_frame, image)

            diff=np.sum(_diff)/previous_frame.shape[0]/previous_frame.shape[1]/3.

            if diff>self.diff_thres:
                return True
            else:
                return False

    def sort_and_filter(self,bboxes):
        '''
        find the top_k max bboxes, and filter the small face

        :param bboxes:
        :return:
        '''

        if len(bboxes)<1:
            return []


        area=(bboxes[:,2]-bboxes[:,0])*(bboxes[:,3]-bboxes[:,1])
        select_index=area>self.min_face

        area=area[select_index]
        bboxes=bboxes[select_index,:]
        if bboxes.shape[0]>self.top_k:
            picked=area.argsort()[-self.top_k:][::-1]
            sorted_bboxes=[bboxes[x] for x in picked]
        else:
            sorted_bboxes=bboxes
        return np.array(sorted_bboxes)

    def judge_boxs(self,previuous_bboxs,now_bboxs):
        '''
        function used to calculate the tracking bboxes

        :param previuous_bboxs:[[x1,y1,x2,y2],... ]
        :param now_bboxs: [[x1,y1,x2,y2],... ]
        :return:
        '''
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

        for i in range(now_bboxs.shape[0]):
            contain = False
            for j in range(previuous_bboxs.shape[0]):
                if iou(now_bboxs[i], previuous_bboxs[j]) > self.iou_thres:
                    result.append(self.smooth(now_bboxs[i],previuous_bboxs[j]))
                    contain=True
                    break
            if not contain:
                result.append(now_bboxs[i])


        return np.array(result)

    def smooth(self,now_box,previous_box):

        return self.filter(now_box[:4], previous_box[:4])






    def reset(self):
        '''
        reset the previous info used foe tracking,

        :return:
        '''
        self.track_box = None
        self.previous_image = None
        self.previous_box = None


