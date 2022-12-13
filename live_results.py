import cv2
from screeninfo import get_monitors
import numpy as np
import random
from pathlib import Path
import csv
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models
from torchvision.transforms import ToTensor

import tqdm
import matplotlib.pyplot as plt
import os



beautiful_eyes_model = torchvision.models.resnet18(weights = "DEFAULT")
beautiful_eyes_model.fc = nn.Linear(512,4, bias=True)

beautiful_eyes_model.load_state_dict(torch.load("C:/Users/hugob/Desktop/Projet INF573/eye_detection/ourmodel2.pth"))
beautiful_eyes_model.eval()



class ImageGenerator:
    def __init__(self, circle_size: int = 10) -> None:
        self.screen_width = get_monitors()[0].width-50
        self.screen_height = get_monitors()[0].height-50
        self.image = np.full((self.screen_height, self.screen_width, 3), 255, np.uint8)
        self.videoCapture = self.launch()
        self.point_size = circle_size
        self.next_id = 0
        self.data_folder = Path("data")
        self.image_folder = Path("data/images")
        self.dataset_name = Path("dataset.csv")

    def launch(self):
        cv2.namedWindow("Dataset Generator")
        videoCapture = cv2.VideoCapture(1)
        if not videoCapture.isOpened():
            raise Exception("Starting video capture failed")
        return videoCapture

    def show_random_point(self):
        x, y = (
            random.randint(self.point_size, self.screen_width - self.point_size-20),
            random.randint(self.point_size, self.screen_height - self.point_size-20),
        )
        self.image = np.full((self.screen_height, self.screen_width, 3), 255, np.uint8)
        self.image = cv2.circle(self.image, (x, y), self.point_size, (255, 0, 0), -1)
        cv2.imshow("Dataset Generator", self.image)
        return x, y
    
    def show_where_you_look(self):
        ret, frame = self.videoCapture.read()
        ret, frame = self.videoCapture.read()
        if not ret:
            return
        image = torch.tensor([np.array(np.transpose(frame, (2, 0, 1)),dtype = np.float32)])


        ypred = beautiful_eyes_model(image)[0]
        maxi = ypred[0]
        maxindex = 0
        for j in range(4):
            if ypred[j]>maxi:
                maxi = ypred[j]
                maxindex = j

        if maxindex == 0:
            x, y = ( self.screen_width //2, 100)
            print("TOP")
        elif maxindex == 1:
            x, y = ( self.screen_width - 100, self.screen_height//2)
            print("RIGHT")
        elif maxindex == 2:
            x, y = ( self.screen_width //2, self.screen_height - 100)
            print("BOTTOM")
        else:
            x, y = ( 100, self.screen_height//2)
            print("LEFT")
            
        
        
        self.image = np.full((self.screen_height, self.screen_width, 3), 255, np.uint8)
        self.image = cv2.circle(self.image, (x, y), self.point_size, (0, 0, 255), -1)
        cv2.imshow("Dataset Generator", self.image)
        return x, y

    def show_instructions(self):
        self.image = cv2.putText(
            self.image,
            "Press Enter to save the image ; Press Esc to exit ; Press any key to continue",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Dataset Generator", self.image)

    def run(self):
        self.show_instructions()
        cv2.waitKey(0)
        x, y = self.show_random_point()
        while True:
            key = cv2.waitKey(20)
            if key == 27:
                break
            elif key == 13:
                x, y = self.show_where_you_look()

        self.csvfile.close()
        self.videoCapture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("launching dataset generation")
    print("please make sure your eyes are in the cameras viewpoint")

    generator = ImageGenerator()
    generator.run()