#-*-coding:utf-8-*-
import tensorflow as tf
import cv2
import numpy as np
import time

from config import config as cfg


class Keypoints:

    def __init__(self):
        self.model_path=cfg.KEYPOINTS.model_path
        self.min_face=20
        self.keypoint_num=cfg.KEYPOINTS.p_num*2

        self._graph = tf.Graph()

        with self._graph.as_default():

            self._graph, self._sess = self.init_model(self.model_path)
            self.img_input = tf.get_default_graph().get_tensor_by_name('tower_0/images:0')
            self.embeddings = tf.get_default_graph().get_tensor_by_name('tower_0/prediction:0')
            self.training = tf.get_default_graph().get_tensor_by_name('training_flag:0')

            self.keypoints=self.embeddings[:,:self.keypoint_num]
            self.headpose=self.embeddings[:,-7:-4]*90.
            self.state=tf.nn.sigmoid(self.embeddings[:,-4:])

    def simple_run(self,cropped_img):

        with self._graph.as_default():

            cropped_img=np.expand_dims(cropped_img,axis=0)

            _keypoints,p,_states = self._sess.run([self.keypoints,self.headpose,self.state], \
                                                    feed_dict={self.img_input: cropped_img, \
                                                               self.training: False})

        return _keypoints,_states

    def run(self,img,bboxes):
        #### should be batched process
        #### but process one by one, more simple



        landmark_result=[]
        state_result=[]
        for i,box in enumerate(bboxes):
            landmark,state=self._one_shot_run(img,box,i)
            if landmark is not None:
                landmark_result.append(landmark)
                state_result.append(state)
        return np.array(landmark_result),np.array(state_result)

    def _one_shot_run(self,image,bbox,i):

        ##preprocess
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        if bbox_width<=self.min_face or bbox_height<=self.min_face:
            return None
        add = int(max(bbox_width, bbox_height))
        bimg = cv2.copyMakeBorder(image, add, add, add, add, borderType=cv2.BORDER_CONSTANT,
                                  value=cfg.DATA.pixel_means)
        bbox += add
        ###extend

        one_edge=(1+2*cfg.KEYPOINTS.base_extend_range[0]) * bbox_width

        center=[(bbox[0]+bbox[2])//2,(bbox[1]+bbox[3])//2]

        bbox[0] = center[0]-one_edge//2
        bbox[1] = center[1]-one_edge//2
        bbox[2] = center[0]+one_edge//2
        bbox[3] = center[1]+one_edge//2

        #
        bbox = bbox.astype(np.int)
        crop_image = bimg[bbox[1]:bbox[3], bbox[0]:bbox[2], :]


        h, w, _ = crop_image.shape
        crop_image = cv2.resize(crop_image, (cfg.KEYPOINTS.input_shape[1], cfg.KEYPOINTS.input_shape[0]))

        cv2.imshow('i am watching u * * %d'%i,crop_image)


        crop_image = crop_image.astype(np.float32)


        _keypoints, _state = self.simple_run(crop_image)

        ##recorver

        res = _keypoints[0][:self.keypoint_num].reshape((-1, 2))
        res[:, 0] = res[:, 0] * w / cfg.KEYPOINTS.input_shape[1]
        res[:, 1] = res[:, 1] * h / cfg.KEYPOINTS.input_shape[0]

        LANDMARK = []
        for _index in range(res.shape[0]):
            x_y = res[_index]
            LANDMARK.append([int(x_y[0] * cfg.KEYPOINTS.input_shape[0] + bbox[0] -add),
                             int(x_y[1] * cfg.KEYPOINTS.input_shape[1] + bbox[1] -add)])

        LANDMARK = np.array(LANDMARK, np.float32)
        print(np.round(_state))
        return LANDMARK,_state


    def init_model(self,*args):

        if len(args) == 1:
            use_pb = True
            pb_path = args[0]
        else:
            use_pb = False
            meta_path = args[0]
            restore_model_path = args[1]

        def ini_ckpt():
            graph = tf.Graph()
            graph.as_default()
            configProto = tf.ConfigProto()
            configProto.gpu_options.allow_growth = True
            sess = tf.Session(config=configProto)
            # load_model(model_path, sess)
            saver = tf.train.import_meta_graph(meta_path)
            saver.restore(sess, restore_model_path)

            print("Model restred!")
            return (graph, sess)

        def init_pb(model_path):
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.2
            compute_graph = tf.Graph()
            compute_graph.as_default()
            sess = tf.Session(config=config)
            with tf.gfile.GFile(model_path, 'rb') as fid:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(fid.read())
                tf.import_graph_def(graph_def, name='')

            # saver = tf.train.Saver(tf.global_variables())
            # saver.save(sess, save_path='./tmp.ckpt')
            return (compute_graph, sess)

        if use_pb:
            model = init_pb(pb_path)
        else:
            model = ini_ckpt()

        graph = model[0]
        sess = model[1]

        return graph, sess