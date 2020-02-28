import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

import network
from utils import *

class CNN():
    def __init__(self, num_classes, device):
        super().__init__()

        print ('==> Initialising spatial CNN model')
        # self.model = models.vgg16(pretrained=True, progress=True)        
        self.model = network.resnet152(pretrained=True, num_classes=num_classes, channels=3)
        
        # Send the model to GPU
        self.model = self.model.to(device)
            
        self.criterion = nn.CrossEntropyLoss()
        # change to adam maybe?
        self.optimiser = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1,verbose=True)