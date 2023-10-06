import torch.nn as nn 
from efficientnet_pytorch import EfficientNet
from torch.nn import functional as F

class EfficientNet_(nn.Module):
    def __init__(self):
        super(EfficientNet_, self).__init__()
        self.network = EfficientNet.from_pretrained('efficientnet-b2', in_channels=3)
        self._fc1 = nn.Linear(1000, 6)
        self._fc2 = nn.Linear(1000, 5)
        self._fc3 = nn.Linear(1000, 3)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, c):
        
        x = self.network(x)  # network로 변경
        
        if c == 'd':
            x = F.relu(x)
            x = self._fc1(x)      
            x = self.softmax(x) 
        if c == 'g':
            x = F.relu(x)
            x = self._fc2(x)   
            x = self.softmax(x) 
        if c == 'e':
            x = F.relu(x)
            x = self._fc3(x) 
            x = self.softmax(x) 
        return x