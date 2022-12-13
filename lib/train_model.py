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
from torchvision.transforms import ToTensor

import tqdm
import matplotlib.pyplot as plt
import csv
import os
import cv2

train_csv = "../data/dataset.csv"
train_images = "../data/images"
#val_csv = "/content/sample_data/california_housing_test.csv"

PRMS = {
    'N_train_entries' : 971,
    'batch_size' : 10,
    'N_validation_entries' : 100,

    'img_size' : 224,
}

class Eye_Dataset(Dataset):
    def __init__(self, path,transform=None):
        self.path = path
        self.transform = transform #maybe for resizing and colordistortion for augmentation
        
        self.file = open(path)

        self.lines = self.file.readlines()[1:-2]
        self.Y = [torch.tensor(float(x.split(',')[1])) for x in self.lines]
        
        self.file.close()
        
        self.file = open(path)
        self.imgpath = [x.split(',')[0] for x in self.lines]
        self.file.close()
        
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        image = cv2.imread(f"../data/images/{self.imgpath[idx]}")
        #image = cv2.resize(image,(224,224))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = np.array(np.transpose(image, (2, 0, 1)),dtype = np.float32)
        #print(image)

        if self.transform:
            image = self.transform(image=image)
        return (image,self.Y[idx])

traindataset = Eye_Dataset(train_csv)
#valdataset = Eye_Dataset(train_csv)

trainloader = DataLoader(traindataset, batch_size= PRMS['batch_size'], shuffle=True)
#valloader = DataLoader(valdataset, batch_size= PRMS['batch_size'], shuffle=False,num_workers=4)

beautiful_eyes_model = torchvision.models.resnet18(weights = "DEFAULT")
beautiful_eyes_model.fc = nn.Linear(512,4, bias=True)
#print(beautiful_eyes_model)
#torch.save(beautiful_eyes_model.state_dict(), "C:/Users/hugob/Desktop/Projet INF573/eye_detection/ourmodel2.pth")
beautiful_eyes_model.eval()

#beautiful_eyes_model.classifier[6] = nn.Linear(4096,2, bias=True) #changing last layer to x and y channels

lossfn = nn.CrossEntropyLoss()


def trainer(model,loss_fn=lossfn,epoch=5,rate=1e-3):

    optimizer = torch.optim.Adam(params = model.parameters(), lr = rate)
    # localBatch = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True)
    for i in tqdm.tqdm(range(epoch), desc = "epoch", leave = False):
        for batch,(x,y) in enumerate(trainloader):
            #print(x.shape)
            #print(model)
            #x = x.type(torch.cuda.LongTensor)
            #y = y.type(torch.cuda.LongTensor)
            y_pred = model.forward(x)

            y = y.type(torch.LongTensor)
            #y_pred = y_pred.type(torch.LongTensor)
            
            print(y)
            print(y_pred)

            loss = loss_fn(y_pred,y)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

def avg_distance_error(model):
    distance = 0
    counter = 0
  # dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=1)
    for x, y in tqdm.tqdm(trainloader, desc='DataLoader', leave=False):
        ypred = model.forward(x)[0]
        distance += torch.norm(ypred-y)
        counter +=1
        if counter > 10:
            break
    return distance/(counter*PRMS['batch_size'])

def success_rate(model):
  count = 0
  
  for x, y in tqdm.tqdm(trainloader, desc='DataLoader', leave=False):
    ypred = model.forward(x)[0]
    maxi = ypred[0]
    maxindex = 0
    for j in range(len(ypred)):
      if ypred[j]>maxi:
        maxi = ypred[j]
        maxindex = j
    if maxindex == y[0]:
      count +=1
  return count/PRMS['N_train_entries']
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
    beautiful_eyes_model.load_state_dict(torch.load("C:/Users/hugob/Desktop/Projet INF573/eye_detection/ourmodel2.pth"))
    #beautiful_eyes_model.fc = nn.Linear(512,4, bias=True)
    beautiful_eyes_model.eval()

    print("model loaded")

    #print(avg_distance_error(model = beautiful_eyes_model))
    trainer(model = beautiful_eyes_model, loss_fn = lossfn, epoch = 1, rate = 1e-5)

    #print(avg_distance_error(model = beautiful_eyes_model))

    print("training finished")
    torch.save(beautiful_eyes_model.state_dict(), "C:/Users/hugob/Desktop/Projet INF573/eye_detection/ourmodel2.pth")
    print("model saved")

