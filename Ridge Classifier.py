
import sklearn.preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.lda import LDA
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

#
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.25)

 
#Dimensionality Reduction

pca = PCA(n_components=50)
pca.fit(X)

X_train_pca = pca.transform(X)

clf = RidgeClassifier()


# Building Nueral Net Model

from sklearn.model_selection import GridSearchCV


param_grid = dict(#n_estimators=[25,50,100,150,175,200,225,250],
                  alpha = [0.5,0.75,1,1.25,1.50,1.75,2,2.25],
                  solver=['svd','cholesky', 'sag'])
                  #max_iter=[250,500,750,1000])
               #algorithm = ['kd_tree', 'ball_tree'] )

grid_search_cv = GridSearchCV(estimator = clf,
                           param_grid = param_grid,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs= 1)



grid_search_cv = grid_search_cv.fit(X_train_pca, y)
best_parameters = grid_search_cv.best_params_
