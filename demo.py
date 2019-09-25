
import cv2
import time

from lib.core.api.facer import FaceAna
from lib.core.headpose.pose import get_head_pose, line_pairs


facer = FaceAna()

def video(video_path_or_cam):


    vide_capture=cv2.VideoCapture(video_path_or_cam)

    while 1:

        ret, img = vide_capture.read()

        img_show = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        star=time.time()
        boxes, landmarks, states = facer.run(img)

        duration=time.time()-star
        print('one iamge cost %f s'%(duration))


        for face_index in range(landmarks.shape[0]):

            #######head pose
            reprojectdst, euler_angle=get_head_pose(landmarks[face_index],img_show)

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


        cv2.namedWindow("capture", 0)
        cv2.imshow("capture", img_show)
        key=cv2.waitKey(1)
        if key==ord('q'):
            return

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Start train.')
    parser.add_argument('--video', dest='video', type=str, default=None, \
                        help='the num of the classes (default: 100)')
    parser.add_argument('--cam_id', dest='cam_id', type=int, default=0, \
                        help='the camre to use')
    args = parser.parse_args()


    if args.video is not None:
        video(args.video)
    else:
        video(args.cam_id)

