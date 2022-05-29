
from random import betavariate
import cv2
import time
import numpy as np
from code import interact
from cv2 import line
import numpy as np

car_classifier = cv2.CascadeClassifier('haarcascade_car.xml')
vid_capture = cv2.VideoCapture('video.avi')

while True:
    ret, frames = vid_capture.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    cars = car_classifier.detectMultiScale(gray, 1.1, 1)
    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow('video', frames)
    if cv2.waitKey(33) == 27:
        break
cv2.destroyAllWindows()


