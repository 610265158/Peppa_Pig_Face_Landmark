
import cv2
import time
import imageio
import numpy as np


from lib.core.api.facer import FaceAna

from lib.core.headpose.pose import get_head_pose, line_pairs
facer = FaceAna()



def video(video_path_or_cam):
    start_holder = False
    vide_capture=cv2.VideoCapture(video_path_or_cam)
    buff=[]
    counter=1
    while 1:
        counter+=1


        ret, image = vide_capture.read()
        #image = image[100:1000, :, :]
        image=cv2.resize(image, None, fx=2., fy=2.)
        pattern = np.zeros_like(image)

        img_show = image.copy()
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        star=time.time()
        boxes, landmarks, states = facer.run(img)

        duration=time.time()-star
        print('one iamge cost %f s'%(duration))


        for face_index in range(landmarks.shape[0]):

            #######head pose
            reprojectdst, euler_angle=get_head_pose(landmarks[face_index],img_show)

            if args.mask:
                face_bbox_keypoints = np.concatenate(
                    (landmarks[face_index][:17, :], np.flip(landmarks[face_index][17:27, :], axis=0)), axis=0)

                pattern = cv2.fillPoly(pattern, [face_bbox_keypoints.astype(np.int)], (1., 1., 1.))
            for start, end in line_pairs:
                cv2.line(img_show, reprojectdst[start], reprojectdst[end], (0, 0, 255),2)

            cv2.putText(img_show, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), thickness=2)
            cv2.putText(img_show, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), thickness=2)
            cv2.putText(img_show, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), thickness=2)

            for landmarks_index in range(landmarks[face_index].shape[0]):

                x_y = landmarks[face_index][landmarks_index]
                cv2.circle(img_show, (int(x_y[0]), int(x_y[1])), 3,
                           (222, 222, 222), -1)








        img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
        img_show=cv2.resize(img_show,None,fx=0.4,fy=0.4)


        if start_holder:
            buff.append(img_show)
        if args.mask:
            cv2.namedWindow("masked", 0)
            cv2.imshow("masked", img * pattern)

        cv2.namedWindow("capture", 0)
        cv2.imshow("capture", img_show)
        key=cv2.waitKey(1)

        if key == ord('s'):
            start_holder=True
        if key==ord('q'):
            buff=[x for i,x in  enumerate(buff) if i%2==0 ]
            gif = imageio.mimsave('sample.gif', buff, 'GIF', duration=0.08)
            return

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Start train.')
    parser.add_argument('--video', dest='video', type=str, default=None, \
                        help='the camera id (default: 0)')
    parser.add_argument('--cam_id', dest='cam_id', type=int, default=0, \
                        help='the camera to use')
    parser.add_argument('--mask', dest='mask', type=bool, default=False, \
                        help='mask the face or not')
    args = parser.parse_args()


    if args.video is not None:
        video(args.video)
    else:
        video(args.cam_id)

