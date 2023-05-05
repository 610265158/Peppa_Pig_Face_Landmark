
import cv2
import time
import numpy as np
import os
import argparse


from Skps import FaceAna

def video(video_path_or_cam):
    facer = FaceAna()
    vide_capture=cv2.VideoCapture(video_path_or_cam)

    while 1:

        ret, image = vide_capture.read()

        if ret:
            pattern = np.zeros_like(image)

            img_show = image.copy()

            star=time.time()
            result = facer.run(image)

            duration=time.time()-star
            #print('one iamge cost %f s'%(duration))

            fps=1/duration
            cv2.putText(img_show, "X: " + "{:7.2f}".format(fps), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), thickness=2)

            for face_index in range(len(result)):

                #######head pose need develop
                #reprojectdst, euler_angle=get_head_pose(landmarks[face_index],img_show)

                cur_face_kps=result[face_index]['kps']
                cur_face_kps_score=result[face_index]['scores']
                for landmarks_index in range(cur_face_kps.shape[0]):

                    x_y = cur_face_kps[landmarks_index]
                    score=cur_face_kps_score[landmarks_index]
                    # color = (255, 255, 255)
                    if score>0.8:
                        color=(255,255,255)
                    else:
                        color = (0, 0, 255)
                    cv2.circle(img_show, (int(x_y[0]), int(x_y[1])),
                                   color=color, radius=1, thickness=2)


            cv2.namedWindow("capture", 0)
            cv2.imshow("capture", img_show)

            if args.mask:
                cv2.namedWindow("masked", 0)
                cv2.imshow("masked", image*pattern)

            key=cv2.waitKey(1)
            if key==ord('q'):
                return


def images(image_dir):
    facer = FaceAna()

    image_list=os.listdir(image_dir)
    image_list=[x for x in image_list if 'jpg' in x or 'png' in x]
    image_list.sort()


    for image_name in image_list:

        image=cv2.imread(os.path.join(image_dir,image_name))

        pattern = np.zeros_like(image)

        img_show = image.copy()

        star=time.time()
        result = facer.run(image)

        ###no track
        facer.reset()

        duration=time.time()-star
        print('one iamge cost %f s'%(duration))

        for face_index in range(len(result)):

            #######head pose
            #reprojectdst, euler_angle=get_head_pose(landmarks[face_index],img_show)

            cur_face_kps = result[face_index]['kps']
            cur_face_kps_score = result[face_index]['scores']
            for landmarks_index in range(cur_face_kps.shape[0]):

                x_y = cur_face_kps[landmarks_index]
                score = cur_face_kps_score[landmarks_index]
                # color = (255, 255, 255)
                if score > 0.8:
                    color = (255, 255, 255)
                else:
                    color = (0, 0, 255)
                cv2.circle(img_show, (int(x_y[0]), int(x_y[1])),
                           color=color, radius=1, thickness=2)

        cv2.namedWindow("capture", 0)
        cv2.imshow("capture", img_show)


        key=cv2.waitKey(0)
        if key==ord('q'):
            return




def build_argparse():
    parser = argparse.ArgumentParser(description='Start train.')
    parser.add_argument('--video', dest='video', type=str, default=None, \
                        help='the camera id (default: 0)')
    parser.add_argument('--cam_id', dest='cam_id', type=int, default=0, \
                        help='the camera to use')
    parser.add_argument('--img_dir', dest='img_dir', type=str, default=None, \
                        help='the images dir to use')

    parser.add_argument('--mask', dest='mask', type=bool, default=False, \
                        help='mask the face or not')



    args = parser.parse_args()
    return  args

if __name__=='__main__':




    args=build_argparse()


    if args.img_dir is not None:
        images(args.img_dir)

    elif args.video is not None:
        video(args.video)
    else:
        video(args.cam_id)

