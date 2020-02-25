import torch
import torch.nn as nn
import glob
import torchvision.models as models
import torch.optim as optim

from utils import *

class CNN():
    def __init__(self, num_classes, device):
        super().__init__()

        print ('==> Initialising temporal CNN model')
        self.model = models.vgg16(pretrained=True, progress=True)
        
        # Reshape the outputs of the last FC layer for 9 classes
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)
        
        # Send the model to GPU
        self.model = self.model.to(device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimiser = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1,verbose=True)