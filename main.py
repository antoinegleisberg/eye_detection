import cv2

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("preview")

##

import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
cap = cv2.VideoCapture(0) # video capture source camera (Here webcyam of laptop)
ret,frame = cap.read() # return a single frame in variable `frame`

background = cv2.imread('C:/Users/hugob/Desktop/Projet INF573/eye_detection/screenalacon2.png')
cv2.imshow("hi",background)
print(background.shape)

while(True):

    id = datetime.today().strftime('%Y%m%d%H%M%S')
    if id != datetime.today().strftime('%Y%m%d%H%M%S'):
        #plt.close("all")
        #cv2.imwrite(f'C:/Users/hugob/Desktop/Projet INF573/eye_detection/captures/{id}.png',frame)
        cap = cv2.VideoCapture(0) # video capture source camera (Here webcyam of laptop)
        ret,frame = cap.read()
        img = cv2.circle(background, (10,10), 1, (255,0,0), -1)
        print(img.shape)
        #break
        cv2.imshow("houhou",img)
        break
        #plt.plot(np.random.randint(0,10),np.random.randint(0,10),'bo')
        #plt.show(block = False)
        print('dingding')



    cv2.imshow('img1',frame) #display the captured image
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC #save on pressing 'y'
        cv2.imwrite(f'C:/Users/hugob/Desktop/Projet INF573/eye_detection/captures/{id}.png',frame)
        cv2.destroyAllWindows()
        print('exited process')
        break
#cap.release()