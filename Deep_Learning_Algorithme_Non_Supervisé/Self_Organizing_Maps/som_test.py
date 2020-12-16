# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:03:07 2019

@author: guill
"""

#Librairies

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importation des données
dataset = pd.read_csv("Credit_Card_Applications.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Changement d'échelle

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Entraînement du SOM
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15)

som.random_weights_init(X)
som.train_random(X,num_iteration=100)

# Visualisation des résultats
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()

markers = ("o", "s")
colors = ("r", "g")

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5,w[1]+0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = "None",
         markersize = 10,
         markeredgewidth = 2)
show()

# Détecter la fraude
mappings = som.win_map(X)
frauds =np.concatenate((mappings[(3,10)], mappings [(7,5)], mappings[(7,7)]),
                                 axis=0)
frauds = sc.inverse_transform(frauds)

    