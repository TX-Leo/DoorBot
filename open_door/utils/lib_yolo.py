from ultralytics import YOLO
import torch
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_model = YOLO("cfg/yolov5s.pt")
image = Image.open('/media/datadisk10tb/leo/projects/realman-robot/open_door/data/test/001.png').convert("RGB")

yolo_model = yolo_model.to(device)

model_vars = vars(yolo_model)
for name, value in model_vars.items():
    print(f'name: {name}, type: {type(value)}')

detections = yolo_model(image)
# print(f'detections:\n{detections}')
print(f'type of detections: {type(detections)}')
detection = detections[0]
print(f'type of detection: {type(detection)}')
boxes = detection.boxes
print(f'type of xywh: {type(boxes)}')
xywh = boxes.xywh
print(f'type of xywh: {type(xywh)}')
print(f'xywh:\n{xywh}')