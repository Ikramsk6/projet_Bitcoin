import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import random as rd 
from  datetime import datetime
from time import time 
from collections import Counter


#**********************************************************
# lecture des données et calculs des indicateurs finaciers 
#**********************************************************


# Fonction pour calculer les moyennes mobiles (SMA et EMA)
def ajouter_moyennes_mobiles(data, periode=5):
    data['SMA'] = data.iloc[:, 4].rolling(window=periode).mean()  # SMA
    data['EMA'] = data.iloc[:, 4].ewm(span=periode, adjust=False).mean()  # EMA

# Fonction pour calculer l'indice de force relative (RSI)
def ajouter_RSI(data, periode=14):
    delta = data.iloc[:, 4].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periode).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periode).mean()
    RS = gain / loss
    data['RSI'] = 100 - (100 / (1 + RS))

def ajouter_bollinger_bands(data, periode=20):
    data['BB_Middle'] = data.iloc[:, 4].rolling(window=periode).mean()
    data['BB_Upper'] = data['BB_Middle'] + (2 * data.iloc[:, 4].rolling(window=periode).std())
    data['BB_Lower'] = data['BB_Middle'] - (2 * data.iloc[:, 4].rolling(window=periode).std())


# Lecture des données et ajout des indicateurs financiers
def LireDonneesAvecIndicateurs(NomFichier):
    RawData = pd.read_csv(NomFichier, sep=';')
    ajouter_moyennes_mobiles(RawData, periode=5)
    ajouter_RSI(RawData, periode=14)
    ajouter_bollinger_bands(RawData, periode=20)

    RawC = []
    for index in range(len(RawData) - 1):
        value = RawData.iloc[index, 4] > RawData.iloc[index + 1, 4]
        value = 1 * value
        RawC.append(value)
    return RawData, RawC

# Mise à jour de CreerData pour inclure les indicateurs financiers
def CreerDataAvecIndicateurs(RawData, RawC, h=5):
    n = len(RawC)
    D = []
    C = []

    toNormalize = []
    unchangedData = []
    for t in range(n - h):
        ouverture = RawData.iloc[t:t+h+1, 1].tolist()
        cloture = RawData.iloc[t:t+h+1, 4].tolist()
        plusHaut = RawData.iloc[t:t+h+1, 2].tolist()
        plusBas = RawData.iloc[t:t+h+1, 3].tolist()
        jourSemaine = jour_semaine(RawData.iloc[t+h, 0])

        # Ajouter les indicateurs financiers pour les dernières valeurs
        SMA = RawData.iloc[t+h, -5]
        EMA = RawData.iloc[t+h, -4]
        RSI = RawData.iloc[t+h, -3]
        BB_Upper = RawData.iloc[t+h, -2]
        BB_Lower = RawData.iloc[t+h, -1]

        toNormalize.append(ouverture + cloture + plusHaut + plusBas + [SMA, EMA, RSI, BB_Upper, BB_Lower])
        unchangedData.append([jourSemaine])
        C.append(RawC[t+h])
        
    toNormalize = np.array(toNormalize)
    medianes = getMedian(toNormalize)
    arr_centre = toNormalize - medianes
    max_abs = getMaxValues(arr_centre)
    normalized = arr_centre / max_abs  

    D = np.concatenate((normalized, np.array(unchangedData)), axis=1)
    C = np.array(C)
    return D, C


def jour_semaine(date):
    date = datetime.strptime(date, "%d/%m/%Y")
    return date.weekday() + 1

medianes = []
def getMedian(bitCoinValues):
    global medianes
    if len(medianes) == 0:
        medianes = np.median(bitCoinValues, axis=0)
    return medianes

maxValues = []
def getMaxValues(bitCoinValues):
    global maxValues
    if len(maxValues) == 0:
        maxValues = np.max(np.abs(bitCoinValues), axis=0)
    return maxValues

RawData_train, RawC_train = LireDonneesAvecIndicateurs("Cotations2020.csv")
RawData_test, RawC_test = LireDonneesAvecIndicateurs("Cotations2021.csv")



#******************************************************
#Préparation des données pour l entraînement et le test
#******************************************************

D_train, C_train = CreerDataAvecIndicateurs(RawData_train, RawC_train)
D_test, C_test = CreerDataAvecIndicateurs(RawData_test, RawC_test)


#********************************
#Définition des fonctions de base
#********************************

 

# Fonction d'activation Sigmoid
def Sigmoid(x):
    x = np.clip(x, -500, 500)  # Limite pour éviter les dépassements numériques
    return 1 / (1 + np.exp(-x))

# Dérivée de Sigmoid -pour la rétropropagation
def dSigmoid(x):
    return Sigmoid(x) * (1 - Sigmoid(x))

# Fonction d'activation ReLU
def ReLU(x):
    return np.maximum(0, x)

# Dérivée de ReLU -pour la rétropropagation
def dReLU(x):
    return 1 * (x > 0)



def CrossEntropyLoss(predictions, targets, epsilon=1e-10):
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))



# Fonction H pour le produit scalaire + biais
def H(weights, constant, point):
    return np.sum(weights * point) + constant



# Calcul des gradients pour optimiser les poids et le biais
def Gradient(weights, constant, D_train, C_train):
    grad = []
    for w in range(len(weights)):
        coord_value = 0
        for index in range(len(D_train)):
            x = D_train[index:index+1, :]
            coord_value -= x[0, w] * (C_train[index] - Sigmoid(H(weights, constant, x)))
        grad.append(coord_value)
    
    coord_value = 0
    for index in range(len(D_train)):
        x = D_train[index:index+1, :]
        coord_value -= (C_train[index] - Sigmoid(H(weights, constant, x)))
    grad.append(coord_value)
    return np.array(grad)


# Algorithme de descente de gradient pour optimiser les poids et le biais
def Algorithme_Gradient(D_train, C_train, nb_step=100, initial_alpha=0.1, decay=0.01):
    weights = np.random.random(np.shape(D_train)[1])  # Initialisation des poids
    constant = np.random.random()  # Initialisation du biais
    step = 1
    prev_E = float('inf')

    # Initialisations pour le suivi de la convergence
    fig, ax = plt.subplots(figsize=(10, 5), nrows=1, ncols=1)
    itersteps = []
    iterGap = []
    iterError = []

    while step < nb_step:
        # Calcul de la perte
        E = CrossEntropyLoss(Sigmoid(H(weights, constant, D_train)), C_train)
        
        # Calcul du gradient
        total_grad = Gradient(weights, constant, D_train, C_train)
        norm = np.linalg.norm(total_grad)

        # Critères d'arrêt
        if norm < 1e-5 or abs(E - prev_E) < 1E-6:
            break

        # Calcul du taux d'apprentissage dynamique
        learning_rate = initial_alpha / (1 + decay * step)

        # Mise à jour des poids et du biais
        weights -= (learning_rate * total_grad[:-1]) / norm
        constant -= learning_rate * total_grad[-1] / norm
        prev_E = E

        # Stockage des valeurs pour affichage
        itersteps.append(step - 1)
        iterGap.append(norm)
        iterError.append(E)
        step += 1

    # Affichage de la convergence
    ax.plot(itersteps, iterGap, label='Norme du gradient')
    ax.plot(itersteps, iterError, label='Erreur')
    ax.set_xlabel('step')
    ax.legend()
    plt.show()
    
    return weights, constant

class OptimizerAdam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, weights, grads):
        if self.m is None:
            self.m = np.zeros_like(grads)
            self.v = np.zeros_like(grads)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        weights -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return weights


#*********************************************************************
#Construction et Entraînement du Modèle (Propagation Avant et Arrière)
#*********************************************************************


# Propagation avant : Calcul des prédictions
def forward_propagation(D_train, weights, constant):
    predictions = []
    for x in D_train:
        z = H(weights, constant, x)  # Calcul de la sortie linéaire
        a = Sigmoid(z)  # Application de la fonction d'activation Sigmoid
        predictions.append(a)
    return np.array(predictions)

# Propagation arrière : Mise à jour des poids et du biais
def backward_propagation(D_train, C_train, weights, constant, learning_rate):
    total_grad = Gradient(weights, constant, D_train, C_train)  # Calcul des gradients
    weights -= learning_rate * total_grad[:-1]  # Mise à jour des poids
    constant -= learning_rate * total_grad[-1]  # Mise à jour du biais
    return weights, constant

# Entraînement du modèle avec Adam
def train_model_with_adam(D_train, C_train, nb_epochs=100, learning_rate=0.01):
    weights = np.random.random(np.shape(D_train)[1])  # Initialisation des poids
    constant = rd.random()  # Initialisation du biais
    optimizer = OptimizerAdam(lr=learning_rate)  # Initialisation d'Adam

    for epoch in range(nb_epochs):
        # Propagation avant
        predictions = forward_propagation(D_train, weights, constant)

        # Calcul de la perte
        loss = CrossEntropyLoss(predictions, C_train)

        # Calcul des gradients
        total_grad = Gradient(weights, constant, D_train, C_train)

        # Mise à jour des poids avec Adam
        weights = optimizer.update(weights, total_grad[:-1])  # Mise à jour des poids
        constant -= learning_rate * total_grad[-1]  # Mise à jour du biais (Adam ne s'applique qu'aux poids)

        # Affichage de la perte pour suivre la convergence
        if epoch % 10 == 0 or epoch == nb_epochs - 1:
            print(f"Epoch {epoch + 1}/{nb_epochs}, Loss: {loss:.4f}")

    return weights, constant



#****************************
# Évaluation des Performances
#****************************

# Évaluation des performances
def evaluate_model(D_test, C_test, weights, constant):
    Cpred = []
    for x in D_test:
        z = H(weights, constant, x)  # Calcul de la sortie linéaire
        a = Sigmoid(z)  # Application de la fonction d'activation Sigmoid
        Cpred.append(1 if a > 0.5 else 0)  # Prédiction finale (seuil à 0.5)

    # Calcul des métriques de performance
    GoodPred = sum([1 for i in range(len(C_test)) if C_test[i] == Cpred[i]])
    VraiPositifs = sum([1 for i in range(len(C_test)) if C_test[i] == 1 and Cpred[i] == 1])
    FauxPositifs = sum([1 for i in range(len(C_test)) if C_test[i] == 0 and Cpred[i] == 1])
    VraiNegatifs = sum([1 for i in range(len(C_test)) if C_test[i] == 0 and Cpred[i] == 0])
    FauxNegatifs = sum([1 for i in range(len(C_test)) if C_test[i] == 1 and Cpred[i] == 0])

    # Calcul des taux
    accuracy = 100.0 * GoodPred / len(C_test)
    precision = 100.0 * VraiPositifs / (VraiPositifs + FauxPositifs) if VraiPositifs + FauxPositifs > 0 else 0
    recall = 100.0 * VraiPositifs / (VraiPositifs + FauxNegatifs) if VraiPositifs + FauxNegatifs > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    # Affichage des résultats
    print(f"Taux de bonnes classifications : {accuracy:.2f}%")
    print(f"Précision (Precision) : {precision:.2f}%")
    print(f"Rappel (Recall) : {recall:.2f}%")
    print(f"F1 Score : {f1_score:.2f}%")



weights, constant = train_model_with_adam(D_train, C_train, nb_epochs=100, learning_rate=0.001)

evaluate_model(D_test, C_test, weights, constant)




