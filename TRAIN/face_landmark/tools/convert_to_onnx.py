import sys
sys.path.append('.')


from lib.core.base_trainer.model import COTRAIN


import torch
import torchvision
import argparse
import re



parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,default=None, help='the thres for detect')
args = parser.parse_args()

model_path=args.model

device=torch.device('cpu')
dummy_input = torch.randn(1, 3,128, 128 , device='cpu')

style_model  = COTRAIN(inference=True).to(device)
style_model.eval()
if model_path is not None:

    state_dict = torch.load(model_path,map_location=device)
    # remove saved deprecated running_* keys in InstanceNorm from the checkpoint

    style_model.load_state_dict(state_dict,strict=False)
    style_model.to(device)




### load your weights
style_model.eval()
# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.


torch.onnx.export(style_model,
                  dummy_input,
                  "face_kps.onnx" ,
                  # dynamic_axes={'input': {2: 'height',
                  #                        3: 'width',
                  #                        }},
                  input_names=["input"],
                  output_names=["output"],opset_version=12)