import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

import network
from utils import *

class CNN():
    def __init__(self, lr, num_classes, channels, device):
        super().__init__()
        
        self.lr = lr
        self.start_epoch = 0
        self.accuracy = 0
        
        print ('==> Initialising temporal CNN model')
        # self.model = models.vgg16(pretrained=True, progress=True)       
        self.model = network.resnet152(pretrained=True, num_classes=num_classes, channels=channels)

        # Send the model to GPU
        self.model = self.model.to(device)
        
        self.criterion = nn.CrossEntropyLoss()
        # change to adam maybe?
        self.optimiser = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1,verbose=True)
        
    def load_checkpoint(self, checkpoint_path):
        checkpoint_file_path = f'{checkpoint_path}/{self.name}/temporal'
        
        if os.path.isfile(checkpoint_file_path):
            checkpoint = torch.load(checkpoint_file_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimiser.load_state_dict(checkpoint['optimiser'])
            self.start_epoch = checkpoint['epoch']
            self.accuracy = checkpoint['accuracy']
            
            print(f'==> Loaded temporal model checkpoint {args.name} at epoch {self.start_epoch} with top1 accuracy {self.accuracy}')
        else:
            print(f'==> No checkpoint at {checkpoint_file_path} -- Training model from scratch')