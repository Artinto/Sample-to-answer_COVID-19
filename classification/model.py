import torch
import torch.nn as nn
import torch.nn.functional as func
from torchvision.models import resnet18, resnet34, resnet50, densenet121, densenet161, mobilenet_v2, squeezenet1_1


class BCE(nn.Module):
    def __init__(self):
        super(BCE, self).__init__()
        self.resnet = resnet18()
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1000)
        self.fc1 = nn.Linear(1000, 400)
        self.fc1_drop = nn.Dropout()
        self.fc2 = nn.Linear(400, 84)
        self.fc2_drop = nn.Dropout()
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc1_drop(func.relu(self.fc1(x)))
        x = self.fc2_drop(func.relu(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
        return x

    def init_weights(model):  # 웨이트 초기화
        w = torch.empty(3, 5)
        nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
