import torch
import math
import os
import time
import cv2
import timm
import numpy as np
import pandas as pd
from IPython import display
import random
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import KFold

class Deep3DCNN(nn.Module):
    def __init__(self, num_classes, extract_features=False):
        super(Deep3DCNN, self).__init__()
        
        # First Convolutional block
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 2, 2)) #16 channels in orig
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # self.pool1 = MaxPool3dWithActivations(kernel_size=2, stride=2)
        
        # Second Convolutional block
        # due to memory issues, for conv2, the channels are dropped from 1,32,32,64 to 1, 16, 16, 32
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # self.pool2 = MaxPool3dWithActivations(kernel_size=2, stride=2)

        # Third Convolutional block
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # self.pool3 = MaxPool3dWithActivations(kernel_size=2, stride=2)
        
        # # Fourth Convolutional block
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(128)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # self.pool4 = MaxPool3dWithActivations(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)  
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        

    def forward(self, x, extract_features=False):
        # Convolutional blocks with BatchNorm, ReLU activation, and pooling
        x = self.pool1(nn.ReLU()(self.bn1(self.conv1(x))))
        #print(f"Shape after pool1: {x.shape}")
        x = self.pool2(nn.ReLU()(self.bn2(self.conv2(x))))
        #print(f"Shape after pool2: {x.shape}")
        x = self.pool3(nn.ReLU()(self.bn3(self.conv3(x))))
        #print(f"Shape after pool3: {x.shape}")
        x = self.pool4(nn.ReLU()(self.bn4(self.conv4(x))))
        #print(f"Shape after pool4: {x.shape}")
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        if extract_features: # if to plot feature maps from the last conv layer
            return x
        #print(f"the shape of flattened features maps before fc1 {x.shape}")
        # Fully connected layers with dropout for regularization
        x = nn.ReLU()(self.fc1(x))
        x = nn.Dropout(0.4)(x)
        x = nn.ReLU()(self.fc2(x))
        x = nn.Dropout(0.3)(x)
        x = self.fc3(x)
        # if extract_features: # if to plot feature maps for t-sne then use this last layer
        #     return x
        
        return x