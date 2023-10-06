from torchvision.models import ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
import torch.nn as nn
import torchvision.models as models
from transformers import AutoFeatureExtractor, ResNetForImageClassification
import torch

class ResExtractor(nn.Module):
    def __init__(self, resnetnum='50', pretrained=True):
        super(ResExtractor, self).__init__()
        if resnetnum == '18':
            self.resnet = models.resnet18(pretrained=pretrained)
        elif resnetnum == '34':
            self.resnet = models.resnet34(pretrained=pretrained)
        elif resnetnum == '50':
            self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        elif resnetnum == '101':
            self.resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        elif resnetnum == '152':
            self.resnet = models.resnet152(weights=ResNet152_Weights.DEFAULT)

        self.modules_front = list(self.resnet.children())[:-2]
        self.model_front = nn.Sequential(*self.modules_front)

    def front(self, x):
        return self.model_front(x)

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()

        arch = '152'
        self.encoder = ResExtractor(arch)
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.batch_norm = nn.BatchNorm1d(num_features=2048)
        self.daily_linear = nn.Linear(2048, 6)
        self.gender_linear = nn.Linear(2048, 5)
        self.embel_linear = nn.Linear(2048, 3)
        
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x, c):
        x = self.encoder.front(x)
        x = self.avg_pool(x).squeeze()
        if c == 'd':
            x = self.daily_linear(x)
            x = self.softmax(x)
        elif c == 'g':
            x = self.gender_linear(x)
            x = self.softmax(x)
        elif c == 'e':
            x = self.embel_linear(x)
            x = self.softmax(x)
        
        return x


class ResNetFeedForwardModel(nn.Module):
    def __init__(self):
        super(ResNetFeedForwardModel, self).__init__()
        self._backbone = models.resnet152(weights=ResNet152_Weights.DEFAULT, progress=True)

        self._feature_size = 1000
        self._hidden_size = 2048
        self._dropout_prob = 0.3
        self._class1_size = 6
        self._class2_size = 5
        self._class3_size = 3

        self._drop = nn.Dropout(self._dropout_prob)
        self._classifier1 = nn.Sequential(
            nn.Linear(self._feature_size, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, self._class1_size),
        )

        self._classifier2 = nn.Sequential(
            nn.Linear(self._feature_size, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, self._class2_size),
        )

        self._classifier3 = nn.Sequential(
            nn.Linear(self._feature_size, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, self._class3_size),
        )

    def forward(self, x, c):
        h = self._backbone(x)
        h = self._drop(h)

        if c == 'd':
            h = self._classifier1(h)
        if c == 'g':
            h = self._classifier2(h)
        if c == 'e':
            h = self._classifier3(h)
        return h

class ResidualClassifier(nn.Module) :
    def __init__(self, feature_size, hidden_size, class_size) :
        super(ResidualClassifier, self).__init__()

        self._feature_size = feature_size
        self._hidden_size = hidden_size
        self._class_size = class_size

        self._long = nn.Sequential(
            nn.Linear(self._feature_size, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, self._hidden_size),
        )

        self._short = nn.Linear(self._feature_size, self._hidden_size)
        self._act = nn.ReLU()
        self._classifier = nn.Linear(self._hidden_size, self._class_size)

    def forward(self, x) :
        h1 = self._long(x)
        h2 = self._short(x)

        h = self._act(h1 + h2)
        o = self._classifier(h)
        return o
    
class ResNetResidualConnectionModel(nn.Module):
    def __init__(self):
        super(ResNetResidualConnectionModel, self).__init__()
        self._backbone = models.resnet152(weights=ResNet152_Weights.DEFAULT, progress=True)

        self._feature_size = 1000
        self._hidden_size = 2048
        self._dropout_prob = 0.3
        self._class1_size = 6
        self._class2_size = 5
        self._class3_size = 3
        # self._matching_layer = nn.Linear(100352, 1000)  # Add this line

        self._classifier1 = ResidualClassifier(
            self._feature_size, 
            self._hidden_size, 
            self._class1_size
        )
        self._classifier2 = ResidualClassifier(
            self._feature_size, 
            self._hidden_size, 
            self._class2_size
        )
        self._classifier3 = ResidualClassifier(
            self._feature_size, 
            self._hidden_size, 
            self._class3_size
        )


    def forward(self, x, c):
        h = self._backbone(x)

        if c == 'd':
            h = self._classifier1(h)
        if c == 'g':
            h = self._classifier2(h)
        if c == 'e':
            h = self._classifier3(h)
        return h