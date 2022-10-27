import torch
import torch.nn as nn


class ABIDENet(nn.Module):
    def __init__(self):
        super(ABIDENet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 90), stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv2d(64, 128, kernel_size=(90, 1), stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

    def forward(self, x):
        return self.feature(x)

class ABIDEClassifier(nn.Module):
    def __init__(self):
        super(ABIDEClassifier, self).__init__()
        self.class_classifier = nn.Sequential()
        # self.class_classifier.add_module('c_drop1', nn.Dropout(p=0.6))
        self.class_classifier.add_module('c_fc1', nn.Linear(128, 96))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop2', nn.Dropout(p=0.6))
        self.class_classifier.add_module('c_fc2', nn.Linear(96, 2))


    def forward(self, x):
        x = self.class_classifier(x)
        return x