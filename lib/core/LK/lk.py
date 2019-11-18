#-*-coding:utf-8-*-
from config import config as cfg
import numpy as np
import math

class GroupTrack():
    def __init__(self):
        self.old_frame = None
        self.previous_landmarks_set = None

        self.with_landmark = True
        self.thres=cfg.TRACE.pixel_thres
        self.alpha=cfg.TRACE.smooth_landmark
        self.iou_thres=cfg.TRACE.iou_thres

        if 'ema' in cfg.TRACE.ema_or_one_euro:
            self.filter=EmaFilter(self.alpha)
        else:
            self.filter=OneEuroFilter()


    def calculate(self, img, now_landmarks_set):

        if self.previous_landmarks_set is None or self.previous_landmarks_set.shape[0]==0:
            self.previous_landmarks_set=now_landmarks_set
            result = now_landmarks_set

        else:
            if self.previous_landmarks_set.shape[0]==0:
                return now_landmarks_set
            else:
                result=[]

                for i in range(now_landmarks_set.shape[0]):
                    not_in_flag = True
                    for j in range(self.previous_landmarks_set.shape[0]):
                        if self.iou(now_landmarks_set[i],self.previous_landmarks_set[j])>self.iou_thres:

                            result.append(self.smooth(now_landmarks_set[i],self.previous_landmarks_set[j]))
                            not_in_flag=False
                            break
                    if not_in_flag:
                        result.append(now_landmarks_set[i])




        result=np.array(result)
        self.previous_landmarks_set=result

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


    def smooth(self,now_landmarks,previous_landmarks):

        result=[]
        for i in range(now_landmarks.shape[0]):

            dis = np.sqrt(np.square(now_landmarks[i][0] - previous_landmarks[i][0]) + np.square(now_landmarks[i][1] - previous_landmarks[i][1]))

            if dis < self.thres:
                result.append(previous_landmarks[i])
            else:
                result.append(self.filter(now_landmarks[i], previous_landmarks[i]))

        return np.array(result)


    def do_moving_average(self, p_now, p_previous):
        p = self.alpha * p_now + (1 - self.alpha) * p_previous
        return p




def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.

        self.dx_prev = float(dx0)
        #self.t_prev = float(t0)

    def __call__(self, x,x_prev):

        if x_prev is None:

            return x




        """Compute the filtered signal."""
        t_e = 1

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
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

