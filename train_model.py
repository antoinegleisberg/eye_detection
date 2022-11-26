##README
"""
file used to train a model for eye-tracking
"""

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models

import tqdm
import matplotlib.pyplot as plt
import csv
import os
import cv2

root = ""
train_csv = "data/dataset.csv"
train_images = "data/images"
#val_csv = "/content/sample_data/california_housing_test.csv"

PRMS = {
    'N_train_entries' : 50,
    'batch_size' : 2,
    'N_validation_entries' : 100,

    'img_size' : 224,
}

class Eye_Dataset(Dataset):
    def __init__(self, path,transform=None):
        self.path = path
        self.transform = transform #maybe for resizing and colordistortion for augmentation
        
        self.file = open(path)

        self.lines = self.file.readlines()[2:-2]
        self.Y = [torch.tensor([float(k) for k in x.split(',')[1:]]) for x in self.lines]
        
        self.file.close()
        
        self.file = open(path)
        self.imgpath = [x.split(',')[0] for x in self.lines]
        self.file.close()

        print('init done')
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.imgpath[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)
        
        return (image,self.Y[idx])

traindataset = Eye_Dataset(train_csv)
#valdataset = Eye_Dataset(val_csv)

trainloader = DataLoader(traindataset, batch_size= PRMS['batch_size'], shuffle=False,num_workers=4)
#valloader = DataLoader(valdataset, batch_size= PRMS['batch_size'], shuffle=False,num_workers=4)

beautiful_eyes_model = torchvision.models.resnet18()
beautiful_eyes_model.fc = nn.Linear(512,2, bias=True) #changing last layer to x and y channels

print("model up and ready")

lossfn = nn.MSELoss(size_average=None, reduce=None, reduction='sum')

print("lossfn defined")


def trainer(model,loss_fn=lossfn,epoch=5,rate=1e-3):

    optimizer = torch.optim.Adam(params = model.parameters(), lr = rate)
    # localBatch = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True)
    print('ding')
    for i in range(epoch):
        print('ding1')
        for batch,(x,y) in enumerate(trainloader):
            print('ding2')

            y_pred = model.forward(x)
            loss = loss_fn(y_pred,y)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

print("training function defined")


"""
def avg_distance_error(model):
    distance = 0
  # dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=1)
    for x, y in tqdm.tqdm(valloader, desc='DataLoader', leave=False):
        ypred = model.forward(x)[0]
        distance += torch.norm(ypred-y)
    return distance/PRMS['N_validation_entries']
"""

if __name__ == "__main__":
    print("launching training")

    trainer(model = beautiful_eyes_model, loss_fn = lossfn, epoch = 5, rate = 1e-3)
    print("training finished")
