import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

path = os.listdir('D:/GitHub/Final Year Project/m/m/Dataset/train/')
classes = {'no_cancer':0, 'cancer':1}
import cv2
X = []
Y = []
for cls in classes:
    pth = 'D:/GitHub/Final Year Project/m/m/Dataset/train/'+cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+'/'+j, 0)
        img = cv2.resize(img, (200,200))
        X.append(img)
        Y.append(classes[cls])
X = np.array(X)
Y = np.array(Y)

X_updated = X.reshape(len(X), -1)
xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10,test_size=.20)
#print(xtrain.max(), xtrain.min())
#print(xtest.max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
#print(xtrain.max(), xtrain.min())
#print(xtest.max(), xtest.min())
warnings.filterwarnings('ignore')

lg = LogisticRegression(C=0.1)
lg.fit(xtrain, ytrain)
sv = SVC()
sv.fit(xtrain, ytrain)
nb=GaussianNB()
nb.fit(xtrain,ytrain)
dt=DecisionTreeClassifier()
dt.fit(xtrain,ytrain)
#pred = sv.predict(xtest)
#dec = {0:'no_cancer', 1:'cancer'}
# img = cv2.imread('/Users/User/Desktop/lung_cancer/Data/valid/cancer/000108 (8).png',0)
# img1 = cv2.resize(img, (200,200))
# img1 = img1.reshape(1,-1)/255
# p = sv.predict(img1)

y_pred_sv=sv.predict(xtest)
acc=accuracy_score(ytest,y_pred_sv)
acc_sv=round(acc*100,2)
y_pred_nb=nb.predict(xtest)
acc=accuracy_score(ytest,y_pred_nb)
acc_nb=round(acc*100,2)
y_pred_dt=dt.predict(xtest)
acc=accuracy_score(ytest,y_pred_dt)
acc_dt=round(acc*100,2)
y_pred_lg=lg.predict(xtest)
acc=accuracy_score(ytest,y_pred_lg)
acc_lg=round(acc*100,2)
x=[acc_sv,acc_nb,acc_dt]
y=["svm","naive byers","decision tree"]
fig,ax=plt.subplots()
plt.xlabel('classifiers')
plt.ylabel('accuracy')
width=0.75
ind=np.arange(len(x))
ax.bar(ind,x,width,color='#B86B77')
plt.xticks(ind,y,color='black')
for i,v in enumerate(x):
    ax.text(i,v+1,str(v),color='black',fontweight='bold')
plot_bytes = pickle.dumps(plt.gcf())

# write the bytes object to a file
with open('plot.pickle', 'wb') as f:
    f.write(plot_bytes)
plt.show()
pickle.dump(sv,open("model.pkl","wb"))
