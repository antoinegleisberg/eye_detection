##README
"""
file used to train a model for eye-tracking
"""

## imports
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import tqdm
import matplotlib.pyplot as plt
import csv
import os
import cv2
##

root =
train_csv =
test_csv =

##

class Eye_Dataset(Dataset):
    def __init__(self, path):
        self.path = path

        self.file = open(path)
        self.header = self.file.readline()
        self.X = np.array([torch.tensor([float(k) for k in x.split(',')][0]) for x in self.file.readlines()])
        self.file.close()

        self.file = open(path)
        self.Y = np.array([torch.tensor([float(k) for k in x.split(',')][1]) for x in self.file.readlines()[1:])
        self.file.close()

        self.file = open(path)
        self.imgpath = np.array([torch.tensor(int(x.split(',')[-1])-1) for x in self.file.readlines()[1:]])
        self.file.close()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = cv2.imread(self.imgpath[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return (self.X[idx],self.Y[idx])






