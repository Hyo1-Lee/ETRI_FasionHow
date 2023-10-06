import timm
import torch.nn as nn
from torch.nn import functional as F


class Rexnet(nn.Module):
    def __init__(self):
        super(Rexnet, self).__init__()
        self.model = timm.create_model('rexnet_200', pretrained=True)
        self._fc1 = nn.Linear(1000, 6)
        self._fc2 = nn.Linear(1000, 5)
        self._fc3 = nn.Linear(1000, 3)
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.model(x)  
        
        d = self._fc1(x)      
        g = self._fc2(x)   
        h = self._fc3(x) 
        
        return d, g, h
 