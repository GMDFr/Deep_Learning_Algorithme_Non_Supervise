# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:00:48 2019

@author: guill
"""
# Hyvrid Architecture

#PART 1 UNSUPERVISED LEARNING


# Self Organizing Map

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(4,5)], mappings[(5,7)]), axis = 0)
frauds = sc.inverse_transform(frauds)

#Partie 2 Passer du non supervisé au supervisé
customers = dataset.iloc[:, 1:].values
y = dataset.iloc[:, -1].values

#Créer la variable indépendante
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i]= 1

# ANN        
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)


# Partie 2
#import keras
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import Dropout

# Initialisation 
classifier = Sequential()
#Ajouter les couches d'entrées et couche cachée
classifier.add(Dense(units=2, activation="relu", 
                     kernel_initializer="uniform", input_dim=15))
#classifier.add(Dropout(rate=0.1))

#Ajout d'une deuxième couche cachée
#classifier.add(Dense(units=8, activation="relu", 
#                     kernel_initializer="uniform"))
#classifier.add(Dropout(rate=0.1))

# Ajout la couche de sortie
classifier.add(Dense(units=1, activation="sigmoid", 
                     kernel_initializer="uniform"))

classifier.compile(optimizer="adam",loss="binary_crossentropy",
                   metrics=["accuracy"])

# Train the neural network
classifier.fit(customers, is_fraud, batch_size=1, epochs=2, verbose=1)

#Predicting the test set results
y_pred = classifier.predict(customers)
        
y_pred = np.concatenate(((dataset.iloc[:,0:1]),y_pred), axis=1)    

y_pred = y_pred[y_pred[:,1].argsort()]