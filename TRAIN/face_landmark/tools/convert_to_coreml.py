import sys
sys.path.append('.')

from lib.core.base_trainer.model import Net,COTRAIN

import argparse

import re

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str,default=None, help='the thres for detect')
args = parser.parse_args()

model_path=args.model

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch

import coremltools as ct
from coremltools.proto import FeatureTypes_pb2 as ft
device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
dummy_input = torch.randn(1, 3,288, 160, device='cpu')
style_model  = COTRAIN(inference=True).to(device)
### load your weights
style_model.eval()



if model_path is not None:
    state_dict = torch.load(model_path,map_location=device)
    # remove saved deprecated running_* keys in InstanceNorm from the checkpoint

    style_model.load_state_dict(state_dict)
    style_model.to(device)
    style_model.eval()
trace = torch.jit.trace(style_model, dummy_input)

# shapes = [(1,3, 128*i, 128*j) for i in range(1, 10) for j in range(1,10)]
# input_shape = ct.EnumeratedShapes(shapes=shapes)

# input_shape = ct.Shape(shape=(1, 3,ct.RangeDim(256, 256*10), ct.RangeDim(256, 256*10)))
# Convert the model
mlmodel = ct.convert(
    trace,
    inputs=[ct.ImageType(name="__input", scale=1/255.,
                         shape=dummy_input.shape)],
minimum_deployment_target=ct.target.iOS14

)




spec = mlmodel.get_spec()

# Edit the spec
ct.utils.rename_feature(spec, '__input', 'image')
ct.utils.rename_feature(spec, 'var_917', 'output')


# to update the shapes of the first input
# input_name = spec.description.input[0].name
# img_size_ranges = ct.models.neural_network.flexible_shape_utils.NeuralNetworkImageSizeRange()
# img_size_ranges.add_height_range((256, -1))
# img_size_ranges.add_width_range((256, -1))

# ct.models.neural_network.flexible_shape_utils.update_image_size_range(spec,
#                                  feature_name=input_name,size_range=img_size_ranges)

# save out the updated model
mlmodel = ct.models.MLModel(spec)
print(mlmodel)
print(spec.description.output)
# for output in spec.description.output:
#     # Change output type
#     output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('RGB')
#     # channels, height, width = tuple(output.type.multiArrayType.shape)
#     #
#     # # Set image shape
#     output.type.imageType.width = 256
#     output.type.imageType.height = 256

# output_name = spec.description.output[0].name
# ct.models.neural_network.flexible_shape_utils.update_image_size_range(spec,
#                                  feature_name=output_name,size_range=img_size_ranges)


mlmodel = ct.models.MLModel(spec)
print(mlmodel)


from coremltools.models.neural_network import quantization_utils
from coremltools.models.neural_network.quantization_utils import AdvancedQuantizedLayerSelector

selector = AdvancedQuantizedLayerSelector(
    skip_layer_types=['batchnorm','bias'],
    minimum_conv_kernel_channels=4,
    minimum_conv_weight_count=4096
)

model_fp8 = quantization_utils.quantize_weights(mlmodel, nbits=8,quantization_mode='linear',selector=selector)



fp_8_file='./neural_style.mlmodel'
model_fp8.save(fp_8_file)