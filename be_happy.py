
import cv2
import time
import imageio

from lib.core.api.facer import FaceAna
from lib.core.LK.lk import GroupTrack

from lib.core.headpose.pose import get_head_pose, line_pairs
facer = FaceAna()



def video(video_path_or_cam):
    start_holder = False
    vide_capture=cv2.VideoCapture(video_path_or_cam)
    buff=[]
    counter=1
    while 1:
        counter+=1

        ret, img = vide_capture.read()

        img = cv2.resize(img, None, fx=1.4, fy=1.4)
        #img = cv2.resize(img, (1280 , 960))
        #img=img[220:1000,:,:]
        img_show = img.copy()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        star=time.time()
        boxes, landmarks, states = facer.run(img)

        duration=time.time()-star
        print('one iamge cost %f s'%(duration))


        for face_index in range(landmarks.shape[0]):

            #######head pose
            reprojectdst, euler_angle=get_head_pose(landmarks[face_index],img_show)

            # for start, end in line_pairs:
            #     cv2.line(img_show, reprojectdst[start], reprojectdst[end], (0, 0, 255),2)

            cv2.putText(img_show, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), thickness=2)
            cv2.putText(img_show, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), thickness=2)
            cv2.putText(img_show, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), thickness=2)

            for landmarks_index in range(landmarks[face_index].shape[0]):

                x_y = landmarks[face_index][landmarks_index]
                cv2.circle(img_show, (int(x_y[0]), int(x_y[1])), 4,
                           (222, 222, 222), -1)

        img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
        img_show=cv2.resize(img_show,None,fx=0.4,fy=0.4)


        if start_holder:
            buff.append(img_show)
        #counter+=1
        # fps=1./duration
        # # cv2.putText(img_show, 'fps %f'%fps, (10, 100),
        # #             cv2.FONT_HERSHEY_SIMPLEX, 1,
        # #             (255, 222, 255), 2)
        # videoWriter.write(img_show)
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
                        help='the num of the classes (default: 100)')
    parser.add_argument('--cam_id', dest='cam_id', type=int, default=0, \
                        help='the camre to use')
    args = parser.parse_args()


    if args.video is not None:
        video(args.video)
    else:
        video(args.cam_id)

