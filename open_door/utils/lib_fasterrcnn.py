import torch
import torchvision
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()

image = Image.open('001.png').convert("RGB")

def transform(image):
    image = F.to_tensor(image)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image

image_tensor = transform(image).unsqueeze(0).to(device)  # add batch dimension

with torch.no_grad():
    detections = model(image_tensor)

# model_vars = vars(model)
# for name, value in model_vars.items():
#     print(f'name: {name}, type: {type(value)}')

print(f'type of detections: {type(detections)}')
for detection in detections:
    print(f'detection:\n{detection}')
    boxes = detection['boxes']
    scores = detection['scores']
    labels = detection['labels']

    print(f'type of boxes: {type(boxes)}')
    print(f'boxes:\n{boxes}')
    print(f'type of scores: {type(scores)}')
    print(f'scores:\n{scores}')
    print(f'type of labels: {type(labels)}')
    print(f'labels:\n{labels}')

    #  Filter the detection results according to the confidence threshold
    high_confidence_indices = scores > 0.5 
    filtered_boxes = boxes[high_confidence_indices]
    # ... Process the filtered bounding box ...