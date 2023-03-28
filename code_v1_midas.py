""">>> --- transform the pytorch checkpoint to the onnx model; """

from midas.model_loader import default_models, load_model
import torch
import cv2
import time
import matplotlib.pyplot as plt


# """>>> +++ load model with pretrained weigths; """
# device = "cpu"
# model_type = "midas_v21_small_256"
# model_path = "/data/SSD1/data/weights/midas_v21_small_256.pt"
# model, transform, net_w, net_h = load_model(
#     device, model_path, model_type=model_type, optimize=False, height=None, square=False
# )
# # print("==>> model: ", model)

# model.eval()


# import torch.onnx

# # Input to the model
# batch_size = 1
# height = 256
# width = 256
# x = torch.randn(batch_size, 3, height, width, requires_grad=True).to(device)

# import os

# os.makedirs("./work_dirs/", exist_ok=True)

# # Export the model
# torch.onnx.export(
#     model,  # model being run
#     x,  # model input (or a tuple for multiple inputs)
#     "./work_dirs/model.onnx",  # where to save the model (can be a file or file-like object)
#     export_params=True,  # store the trained parameter weights inside the model file
#     opset_version=11,  # the ONNX version to use
#     do_constant_folding=True,  # whether to optimize the model by folding constants
#     input_names=["input"],  # the model's input names
#     output_names=["output"],  # the model's output names
#     dynamic_axes={
#         "input": {0: "batch_size"},  # variable-length axes
#         "output": {0: "batch_size"},
#     },
# )


""">>> +++ check the onnx model;"""
import onnxruntime
import numpy as np

# Load the ONNX model
session = onnxruntime.InferenceSession("./work_dirs/model.onnx")

# Input to the model
batch_size = 1
height = 256
width = 256
x = np.random.randn(batch_size, 3, height, width).astype(np.float32)
print("==>> x.shape: ", x.shape)

""">>> +++ prepare input image; """
device = "cpu"
model_type = "midas_v21_small_256"
filename = "./dog.jpg"
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

img = cv2.imread(filename)
img = cv2.resize(img, (width, height))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(device)
print("==>> input_batch.shape: ", input_batch.shape)

input_batch_np = input_batch.numpy()

# Run the model
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
# result = session.run([output_name], {input_name: x})
result = session.run([output_name], {input_name: input_batch_np})
print("==>> result.shape: ", np.array(result).shape)
print("==>> result: ", result)

output = result[0][0]
print("==>> output.shape: ", output.shape)

plt.imshow(result[0][0])
plt.savefig("./midas_code_v2_onnx.png")

