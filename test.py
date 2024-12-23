# from models.yolo import Model

# model = Model(cfg='models/yolov5s.yaml')

from torchvision.models import densenet121
import torch

# Tùy chỉnh lớp cuối cùng
model = densenet121(pretrained=False)
num_features = model.classifier.in_features
model.classifier = torch.nn.Linear(num_features, 10)  # 10 classes
print(model)
