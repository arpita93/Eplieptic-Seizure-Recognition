import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


#import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout

data = pd.read_csv("data.csv")



X= data.iloc[:,1:179].values
y= data.iloc[:,179].values

for i in range(len(y)):
    if y[i] == 1:
        y[i] = 1
    else:
        y[i] = 0
        

# Normalizing data
model= StandardScaler()
X= model.fit_transform(X)
 
#

#Kbest = SelectKBest(k=100)
#X=Kbest.fit_transform(X,y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)

# Building Nueral Net Model
clf= Sequential()

#Adding input layer and first hidden layer
clf.add(Dense(output_dim = 64, activation = 'relu', input_dim = 178))
clf.add(Dropout(0.2))# 25, 25
#Adding second hidden layer
clf.add(Dense(output_dim = 64, activation = 'relu'))
clf.add(Dropout(0.2))

#clf.add(Dense(output_dim = 90, activation = 'relu'))
#clf.add(Dropout(0.25))


clf.add(Dense(1, activation="sigmoid"))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
clf.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


clf.fit(X_train, y_train,
          epochs=50,
          batch_size=16, verbose=1)



y_pred = clf.predict(X_test)

for i in range(len(y_pred)):
    if y_pred[i]>0.5:
        y_pred[i]=1
    else:
        y_pred[i]=0
        
#y_pred= label_binarizer.inverse_transform(y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)

score = metrics.accuracy_score(y_test, y_pred)

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
print ("Classification report: \n", (classification_report(y_test, y_pred)))
print ("F1 weighted averaging:",(f1_score(y_test, y_pred, average='micro')))