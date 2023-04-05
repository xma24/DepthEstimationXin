""">>> +++调用yolov5的包之后嵌入到Python代码里面使用; """
import yolov5

# load pretrained model
model = yolov5.load("yolov5s.pt")

# or load custom model
# model = yolov5.load("train/best.pt")

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image
# image
img = "https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg"
# inference
results = model(img)
# inference with larger input size
results = model(img, size=1280)
# inference with test time augmentation
results = model(img, augment=True)
# parse results
predictions = results.pred[0]
boxes = predictions[:, :4]  # x1, x2, y1, y2
scores = predictions[:, 4]
categories = predictions[:, 5]
# show results
results.show()
# save results
results.save(save_dir="results/")


""">>> +++直接在命令行调用yolov5使用; 
+++这里的0代表的是webcam; """
# yolov5 detect --source 0  # webcam
#                          file.jpg  # image
#                          file.mp4  # video
#                          path/  # directory
#                          path/*.jpg  # glob

# yolov5 detect --source ./yolo_tree_test.jpg


""">>> +++把yolov5的weights给导出来; +++注意--include给出的是一个string的类似tuple的结构,通过,来进行分割; """
# $ yolov5 export --weights yolov5s.pt --include 'torchscript,onnx,coreml,pb,tfjs'
# yolov5 export --weights yolov5s.pt --include 'torchscript,onnx,coreml,pb,tfjs'


""">>> +++可以使用yolov5来做tracking; +++参考网址: https://github.com/tryolabs/norfair/tree/master/demos/yolov5; """


""">>> +++如果输入的图片很大的话,需要使用到slice inference, SAHI这个包可以使用; """

