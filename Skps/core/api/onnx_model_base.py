import cv2
import numpy as np
import onnxruntime as rt
from pathlib import Path
#
class ONNXEngine:
    def __init__(self,onnx_f,device='cpu'):

        self.device = device

        providers = [ 'CPUExecutionProvider' ]
        if 'cuda' in self.device:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider', ]
        self.session = rt.InferenceSession(onnx_f, providers=providers)


    def __call__(self, data):

        ### suport 1 input and 1 output for now


        # Inference
        y_onnx = self.session.run([],
                                  {self.session.get_inputs()[0].name: data})


        return y_onnx


