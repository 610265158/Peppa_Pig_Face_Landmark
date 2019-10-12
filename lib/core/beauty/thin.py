import numpy as np
import cv2
import math



'''
The algorithm is quiet time consume now ,need optimise
'''


class FaceThin():
    def __init__(self):
        pass

    def localTranslationWarp(self,srcImg, startX, startY, endX, endY, radius):

        ddradius = float(radius * radius)
        copyImg = srcImg.copy()


        ddmc = (endX - startX) * (endX - startX) + (endY - startY) * (endY - startY)
        H, W, C = srcImg.shape

        for i in range(W):
            for j in range(H):
                # 计算该点是否在形变圆的范围之内
                # 优化，第一步，直接判断是会在（startX,startY)的矩阵框中
                if math.fabs(i - startX) > radius and math.fabs(j - startY) > radius:
                    continue

                distance = (i - startX) * (i - startX) + (j - startY) * (j - startY)

                if (distance < ddradius):
                    # 计算出（i,j）坐标的原坐标
                    # 计算公式中右边平方号里的部分
                    ratio = (ddradius - distance) / (ddradius - distance + ddmc)
                    ratio = ratio * ratio

                    # 映射原位置
                    UX = i - ratio * (endX - startX)
                    UY = j - ratio * (endY - startY)

                    # 根据双线性插值法得到UX，UY的值
                    value = self.BilinearInsert(srcImg, UX, UY)
                    # 改变当前 i ，j的值
                    copyImg[j, i] = value

        return copyImg



    def BilinearInsert(self,src, ux, uy):
        w, h, c = src.shape
        if c == 3:
            x1 = int(ux)
            x2 = x1 + 1
            y1 = int(uy)
            y2 = y1 + 1

            part1 = src[y1, x1].astype(np.float) * (float(x2) - ux) * (float(y2) - uy)
            part2 = src[y1, x2].astype(np.float) * (ux - float(x1)) * (float(y2) - uy)
            part3 = src[y2, x1].astype(np.float) * (float(x2) - ux) * (uy - float(y1))
            part4 = src[y2, x2].astype(np.float) * (ux - float(x1)) * (uy - float(y1))

            insertValue = part1 + part2 + part3 + part4

            return insertValue.astype(np.int8)



    def __call__(self, image,landmarks):

        if len(landmarks) == 0:
            return

        for landmarks_node in landmarks:
            left_landmark_top = landmarks_node[3]
            left_landmark_down = landmarks_node[5]

            right_landmark_top = landmarks_node[13]
            right_landmark_down = landmarks_node[15]

            endPt = landmarks_node[30]

            # 4-6
            r_left = math.sqrt(
                np.square(left_landmark_top[ 0] - left_landmark_down[0]) +
                np.square(left_landmark_top[ 1] - left_landmark_down[1]) )

            # 14-16
            r_right = math.sqrt(
                np.square(right_landmark_top[0] - right_landmark_down[ 0])  +
                np.square(right_landmark_top[1] - right_landmark_down[ 1] ))

            # left
            thin_image = self.localTranslationWarp(image,
                                                   left_landmark_top[0],
                                                   left_landmark_top[1],
                                                   endPt[0],
                                                   endPt[1],
                                                   r_left)
            # right
            thin_image = self.localTranslationWarp(thin_image,
                                                   right_landmark_top[0],
                                                   right_landmark_top[1],
                                                   endPt[0],
                                                   endPt[1],
                                                   r_right)


        cv2.imshow('thin', thin_image)

