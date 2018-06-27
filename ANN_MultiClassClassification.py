import sklearn.preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from sklearn.lda import LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline


#import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout

data = pd.read_csv("data.csv")



X= data.iloc[:,1:179]
y=data.iloc[:,179]

#X= sklearn.preprocessing.normalize(X,norm='max')   

# Normalizing data
model= StandardScaler()
X= model.fit_transform(X)
 


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)

# For Multiclass Classification, converting output class into  

label_binarizer = sklearn.preprocessing.LabelBinarizer()
label_binarizer.fit(range(max(y_train)+1))




clf= Sequential()


#Adding input layer and first hidden layer
clf.add(Dense(output_dim = 90, activation = 'relu', input_dim = 178))
clf.add(Dropout(0.25))# 25, 25
#Adding second hidden layer
clf.add(Dense(output_dim = 90, activation = 'relu'))
clf.add(Dropout(0.25))

#clf.add(Dense(output_dim = 90, activation = 'relu'))
#clf.add(Dropout(0.25))


clf.add(Dense(6, activation="sigmoid"))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
clf.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


y_train = label_binarizer.transform(y_train)

clf.fit(X_train, y_train, batch_size = 10, nb_epoch = 65)

y_pred = clf.predict(X_test)

y_pred= label_binarizer.inverse_transform(y_pred)

score = metrics.accuracy_score(y_test, y_pred)

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
print ("Classification report: \n", (classification_report(y_test, y_pred)))
print ("F1 weighted averaging:",(f1_score(y_test, y_pred, average='micro')))