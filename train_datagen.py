import cv2
import os
from os import listdir
from os.path import isfile, join
from keras.preprocessing.image import img_to_array
import numpy as np
from sklearn.model_selection import train_test_split

def train_datagen():
  data = []
  labels = []
  dirs =['anger','disgust','fear','happy','sad','surprise']
  for file in dirs:
    dirname = join('/home/chandan/Documents/I. KDEF-dyn I/S4 Stimuli (Video-clips)/',file)
    for img in listdir(dirname):
      img = join(dirname,img)     
      image = cv2.imread(img)
      image = cv2.resize(image,(224,224))
      image = img_to_array(image)
      data.append(image)
      labels.append(file)     
      print img
  data = np.array(data, dtype="float") / 255.0
  labels = np.array(labels)
  return data,labels

data,labels = train_datagen()
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2)
print trainX.shape
np.savetxt('data.txt', trainX, fmt='%s')


x=np.loadtxt('data.txt', dtype=str)
print type(x)
