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

 
#Dimensionality Reduction

pca = PCA(n_components=50)
pca.fit(X)

X_train_pca = pca.transform(X)

clf = SGDClassifier(loss="epsilon_insensitive", penalty="l2", alpha= 0.2, max_iter=1000)

from sklearn.model_selection import GridSearchCV


param_grid = dict(loss = ['hinge','log','modified_huber','squared_hinge','perceptron','squared_loss','huber','epsilon_insensitive'],
                  #random_state= [7,42],
                  # max_features= ['sqrt',
                  alpha = [0.1,0.15,0.2,0.25,0.3],
                  #solver=['svd','cholesky', 'sag'])
                  max_iter=[250,500,750,1000])
               # algorithm = ['kd_tree', 'ball_tree'] )

grid_search_cv = GridSearchCV(estimator = clf,
                           param_grid = param_grid,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs= 1)



grid_search_cv = grid_search_cv.fit(X_train_pca, y)
best_parameters = grid_search_cv.best_params_
