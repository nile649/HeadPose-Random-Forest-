'''
Data feeder :
1. Use media pipe to get face detection.
2. Else use any face detector.
'''
import os
import cv2
from random import shuffle
from tqdm import tqdm
class DataHeadposeDataset:
    def __init__(self, Gabor):
        self.gabor = Gabor()
    def separate(self,text):
        res = []
        sign = True
        start = 0
        for i in range(1,len(text)):
            if text[i]=='+' or text[i]=='-':
                res.append(text[start:i])
                start = i
        res.append(text[start:len(text)])
        ans = []
        for x in res:
            if x[0]=='+':
                ans.append(float(x[1:]))
            else:
                ans.append(-1*float(x[1:]))
        return ans
    def __getImgFeat__(self,text):
        with open(text) as f:
            l = f.readlines()
        img = cv2.imread(text[:-3]+'jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        path = text[:-3]+'jpg'
        cx,cy,xmax,ymax = [int(x) for x in l[3:]]
        xmin = int(cx-xmax/2)
        ymin = int(cy-ymax/2)

        pitch,yaw = self.separate(l[0].split("\n")[0][11:-4])

        return img,[xmin,ymin,xmin+xmax,ymin+ymax],pitch,yaw
    def __data__(self,filelist):
        X = []
        y = []
        for file in (tqdm(filelist, position=2, desc='Image-label list')):
            img,crop,pitch,yaw = self.__getImgFeat__(file)
            face_cropped = img[crop[1]:crop[3],crop[0]:crop[2]]
            face_cropped = cv2.resize(face_cropped, (96,96), interpolation = cv2.INTER_AREA)
            feat = self.gabor.getGabor(face_cropped)
            X.append(feat)
            y.append(yaw)
        return X,y
            

    def __call__(self,pathHeadpose = '../../data/HeadPoseImageDatabase'):
        filelist = []
        for f in os.listdir(pathHeadpose):
            if f=='.DS_Store' or f=='README.txt':
                continue
            path_2 = os.path.join(pathHeadpose,f)
            for ff in os.listdir(path_2):
                if ff=='.DS_Store' or ff[-3:]=='jpg' or ff=='README.txt':
                    continue
                path_3 = os.path.join(path_2,ff)
                filelist.append(path_3)
        shuffle(filelist)
        train_list_headposedb = filelist[:-int(len(filelist)*0.15)]
        test_list_headposedb = filelist[-int(len(filelist)*0.15):]
        X_train,y_train = self.__data__(train_list_headposedb)
        X_test,y_test = self.__data__(test_list_headposedb)
        return X_train,y_train, X_test,y_test
        

