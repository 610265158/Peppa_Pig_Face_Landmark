#-*-coding:utf-8-*-

import numpy as np
import math

class GroupTrack():
    def __init__(self,cfg):
        self.old_frame = None
        self.previous_landmarks_set = None

        self.with_landmark = True
        self.thres=cfg['pixel_thres']
        #self.alpha=cfg.TRACE.smooth_landmark
        self.iou_thres=cfg['iou_thres']

        self.filter=OneEuroFilter()
        # self.filter=EmaFilter(0.5)

    def calculate(self, img, now_landmarks_set):

        h,w,c=img.shape
        scale=[w,h]
        if self.previous_landmarks_set is None or self.previous_landmarks_set.shape[0]==0:
            self.previous_landmarks_set=now_landmarks_set
            previous_dx=np.zeros_like(now_landmarks_set)

            result = now_landmarks_set

        else:

            result=[]
            previous_dx=[]
            for i in range(now_landmarks_set.shape[0]):
                not_in_flag = True
                for j in range(self.previous_landmarks_set.shape[0]):
                    if self.iou(now_landmarks_set[i],self.previous_landmarks_set[j])>self.iou_thres:

                        filtered_res=self.smooth(now_landmarks_set[i]/scale,
                                                 self.previous_landmarks_set[j]/scale,
                                                 self.previous_dx[j]/scale)*scale
                        # filtered_res[stay_indx]=self.previous_landmarks_set[j][stay_indx]
                        result.append(filtered_res)
                        previous_dx.append(self.previous_landmarks_set[j]-filtered_res)
                        not_in_flag=False
                        break

                if not_in_flag:
                    result.append(now_landmarks_set[i])
                    previous_dx.append(np.zeros_like(now_landmarks_set[i]))


        result=np.array(result)
        self.previous_landmarks_set=result
        self.previous_dx=np.array(previous_dx)

        return result

    def iou(self,p_set0,p_set1):
        rec1=[np.min(p_set0[:,0]),np.min(p_set0[:,1]),np.max(p_set0[:,0]),np.max(p_set0[:,1])]
        rec2 = [np.min(p_set1[:, 0]), np.min(p_set1[:, 1]), np.max(p_set1[:, 0]), np.max(p_set1[:, 1])]

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
        intersect = max(0, x2 - x1) * max(0, y2 - y1)

        return intersect / (sum_area - intersect)

    def smooth(self, now_landmarks, previous_landmarks,previous_df):



        filtered_landmarkd=self.filter(now_landmarks, previous_landmarks,previous_df)

        return filtered_landmarkd


    def do_moving_average(self, p_now, p_previous):
        p = self.alpha * p_now + (1 - self.alpha) * p_previous
        return p

def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


def distance(x,y):
    return np.sqrt(np.sum((x-y)**2,axis=1))

class OneEuroFilter:
    def __init__(self, dx0=0.0, min_cutoff=0.15, beta=0.8,
                 d_cutoff=1):
        """Initialize the one euro filter."""
        # The parameters.

        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff



    def __call__(self, x,x_prev,dx_prev):

        """Compute the filtered signal."""
        t_e = 1

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)

        dx = np.sqrt(np.sum((x-x_prev)**2,axis=1))

        ## switch to distance
        dx_prev=np.sqrt(np.sum((dx_prev)**2,axis=1))

        dx_hat = exponential_smoothing(a_d, dx, dx_prev)
        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)

        a = smoothing_factor(t_e, cutoff)

        a=np.expand_dims(a,-1)

        keep_indx=dx<0.002
        a[keep_indx]=0.01

        x_hat = exponential_smoothing(a, x, x_prev)

        # Memorize the previous values.
        self.dx_prev = dx_hat




        return x_hat





class EmaFilter():
    def __init__(self,alpha):
        self.alpha=alpha

    def __call__(self,p_now,p_previous ):
        p=exponential_smoothing(self.alpha,p_now,p_previous)

        return p

