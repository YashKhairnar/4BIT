import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import pandas as pd


model = load_model('D:\GitHub\\4BIT\CTScan\Lung_Model.h5')
print(model.summary())

#img = Image.open('D:\GitHub\\4BIT\CTScan\Dataset2\\test\Bengin case (4).jpg')
#np_img = numpy.array(img)
# np_img = np_img.reshape(224, 224, 3)
#print(model.evaluate(np_img))

image = cv2.imread('D:\GitHub\\4BIT\CTScan\Dataset2\\test\Malignant case (3).jpg')
resized_image = cv2.resize(image , (224 , 224))
input_image = np.expand_dims(resized_image, axis =0)
prediction = model.predict(input_image)
pred = np.argmax(prediction, axis=1) #pick class with highest  probability
print("pred : " , pred)
result = ""
if(prediction[0][0]==1) : 
    result = "Malignant"
elif(prediction[0][1]==1) : 
    result = "Normal"
elif(prediction[0][2]==1) : 
    result = "Benign"

print('Result : ' , result)


print("prediction : " , prediction)