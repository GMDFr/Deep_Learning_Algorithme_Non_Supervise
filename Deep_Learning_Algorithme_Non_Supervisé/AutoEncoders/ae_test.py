# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
#Librairies
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


# Importation du jeu de données
movies = pd.read_csv("ml-1m/movies.dat", 
                     sep="::",
                     header=None,
                     engine="python",
                     encoding="latin-1")

users = pd.read_csv("ml-1m/users.dat", 
                     sep="::",
                     header=None,
                     engine="python",
                     encoding="latin-1")

ratings = pd.read_csv("ml-1m/ratings.dat", 
                     sep="::",
                     header=None,
                     engine="python",
                     encoding="latin-1")

#Préparation du jeu d'entraînement et du jeu de test
training_set = pd.read_csv("ml-100k/u1.base",delimiter="\t", header=None) #  \t = Tab en python
training_set = np.array(training_set, dtype="int")

test_set = pd.read_csv("ml-100k/u1.test",delimiter="\t", header=None) #  \t = Tab en python
test_set = np.array(training_set, dtype="int")

# Obtenir le nombre d'utilisateurs et le nombre de films
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

#Conversion des données en matrice.
def convert(data):
    new_data =[]
    for id_users in range (1, nb_users + 1):
        id_movies = data[data[:,0] == id_users,1]
        id_ratings = data[data[:,0] == id_users,2]
        ratings = np.zeros(nb_movies)
        ratings[id_movies-1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

        
# Conversion des données en tenseurs
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Architecture de l'autoencodeur
class SAE(nn.Module):  # Notion d'héritage
    def __init__(self):
        super(SAE,self).__init__() #Récupère toutes les méthodes contenu dans module
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20,10)
        self.fc3 = nn.Linear(10,20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
        
    def forward(self,x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

sae = SAE()

criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Entraînement de l'auto-encodeur
nb_epochs = 200
for epoch in range(1, nb_epochs + 1):
    train_loss = 0
    s = 0.
    for id_user in range (nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output,target)
            mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10) # corrige l'erreur uniquement les films notés
            loss.backward()
            train_loss += np.sqrt(loss.item() * mean_corrector)
            s += 1.
            optimizer.step() # intensité de la MAJ des poids
    print("epoch: " + str(epoch) + " loss: " + str(train_loss / s))
            
# Test de l'auto-encodeur
test_loss = 0
s = 0.
for id_user in range (nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = Variable(test_set[id_user])
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output,target)
            mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10) # corrige l'erreur uniquement pour les films notés
            loss.backward()
            test_loss += np.sqrt(loss.item() * mean_corrector)
            s += 1. # Compteur
print(" loss: " + str(train_loss / s))            
            
        













