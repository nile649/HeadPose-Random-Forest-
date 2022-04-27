
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

import os
from random import shuffle
from tqdm import tqdm
import joblib
from gabor import GaborData
from facedetector import FaceDetector

class RF:
    def __init__(self):
        self.gabor = GaborData()
        self.loaded_rf = joblib.load("./random_forest.joblib")
        self.getFace = FaceDetector()
    def __getImgFeat__(self,text):
        if isinstance(text,str):
            img = cv2.imread(text)
        else:
            img = text
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        patch = self.getFace(img)
        return patch
    def __data__(self,file):
        img = self.__getImgFeat__(file)
        face_cropped = cv2.resize(img, (96,96), interpolation = cv2.INTER_AREA)
        feat = self.gabor.getGabor(face_cropped)
        return feat

    def __call__(self,pathHeadpose = '../../data/HeadPoseImageDatabase/Person04/person04101-60-90.jpg'):
        feat = self.__data__(pathHeadpose)
        
        res = self.loaded_rf.predict(feat)
        return res


cap = cv2.VideoCapture(0) 
#cap = cv2.VideoCapture('../../pipelineAttention/test_case/test_1.mov')

rf = RF()
while cap.isOpened():
	success, image = cap.read()
	res = 0
	try:
		res = rf(image)
	except:
		continue

	cv2.putText(image, str(res), (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
	cv2.imshow("Head Pose", image)

	if cv2.waitKey(5) & 0xFF == 27:
		break

cap.release()
