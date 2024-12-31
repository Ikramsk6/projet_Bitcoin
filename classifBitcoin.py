import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from datetime import datetime
from time import time


#Input: nom du fichier .csv à lire.
#Nécessite :les lignes du fichier sont classées par dates de cotation croissantes et les colonnes séparées par ";"
#Output: RawData, contient l'historique des cotations telles que lues dans le .csv
#        RawC, contient pour chaque ligne/date t (sauf la dernière) 1 si le cours baisse à t+1; 0 sinon.
def LireDonnees(NomFichier):
    #Lecture des données dans le fichier csv
    RawData = pd.read_csv(NomFichier, sep=';')
    RawC = []
    for index in range(len(RawData) - 1):
        value = RawData.iloc[index, 4] > RawData.iloc[index + 1, 4]
        value = 1 * value
        RawC.append(value)
    return RawData, RawC

# Séparation des données en apprentissage et test
#RawData_train, RawC_train = LireDonnees("Cotations2020.csv")  # Données d'apprentissage
#RawData_test, RawC_test = LireDonnees("Cotations2021.csv")    # Données de test


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

#Input: RawData et RawC ont été créés à partir d'un fichier .csv et h est le nombre de jours utilisé pour la prédiction
#Output: D, contient les points $(\vec x)$. Cela doit être une matrice numpy
#        C, contient la classe d'appartenance $c_x$ de chaque entrée dans data. Cela doit être un vecteur numpy
def CreerData(RawData, RawC, h=5):
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

        toNormalize.append(ouverture + cloture + plusHaut + plusBas)
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

# Création des données pour l'apprentissage et le test
#D_train, C_train = CreerData(RawData_train, RawC_train)  # Données d'apprentissage
#D_test, C_test = CreerData(RawData_test, RawC_test)      # Données de test


# définition de la méthode pour calculer H(v)
def H(weights, constant, point):
    value = np.sum(weights * point) + constant
    return value

# définition de la fonction sigmoïde (logistique)
def Sigmoid(x, lamda=0.1):
    return 1 / (1 + np.exp(-lamda * x))

# définition de la méthode pour calculer l'erreur 
def Loss_function(weights, constant):
    loss_value = 0
    for index in range(len(D_train)):
        p = D_train[index:index+1, :]
        c = C_train[index]
        loss_value -= c * np.log2(Sigmoid(H(weights, constant, p))) + \
                      (1 - c) * np.log2(1 - Sigmoid(H(weights, constant, p)))
    return loss_value

# définition du calcul de gradient
def Gradient(weights, constant):
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

# Algorithme de descente de gradient pour apprentissage
def Algorithme_Gradient():
    nb_step = 100
    initial_alpha = 0.1
    decay = 0.01

    weights = np.random.random(np.shape(D_train)[1])
    constant = rd.random()
    step = 1
    prev_E = float('inf')

    # Initialisations pour graphiques
    fig, ax = plt.subplots(figsize=(10, 5), nrows=1, ncols=1)
    itersteps = []
    iterGap = []
    iterError = []

    while step < nb_step:
        E = Loss_function(weights, constant)
        total_grad = Gradient(weights, constant)
        norm = np.linalg.norm(total_grad)

        if norm < 1e-5 or abs(E - prev_E) < 1E-6:
            break

        learning_rate = initial_alpha / (1 + decay * step)
        weights -= (learning_rate * total_grad[:-1]) / norm
        constant -= learning_rate * total_grad[-1] / norm
        prev_E = E

        itersteps.append(step - 1)
        iterGap.append(norm)
        iterError.append(E)
        step += 1

    ax.plot(itersteps, iterGap, label='Norme du gradient')
    ax.plot(itersteps, iterError, label='Erreur')
    ax.set_xlabel('step')
    ax.legend()
    plt.show()
    
    return weights, constant

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



# Redéfinir PredictionsOnBase
def PredictionsOnBase(weights, constant, D):
    Cpred = []
    for index in range(len(D)):
        if Sigmoid(H(weights, constant, D[index:index+1, :])) > 0.5:
            Cpred.append(1)
        else:
            Cpred.append(0)
    return Cpred

# Lecture des données avec indicateurs financiers
RawData_train, RawC_train = LireDonneesAvecIndicateurs("Cotations2020.csv")
RawData_test, RawC_test = LireDonneesAvecIndicateurs("Cotations2021.csv")

# Création des jeux de données avec indicateurs
D_train, C_train = CreerDataAvecIndicateurs(RawData_train, RawC_train)
D_test, C_test = CreerDataAvecIndicateurs(RawData_test, RawC_test)

# Apprentissage
weights, constant = Algorithme_Gradient()

# Prédictions sur les données de test
Cpred_test = PredictionsOnBase(weights, constant, D_test)

# Évaluation des performances
GoodPred = sum([1 for i in range(len(C_test)) if C_test[i] == Cpred_test[i]])
print("Taux de bonnes classifications : ", 100.0 * GoodPred / len(C_test), "%")


"""
# Apprentissage
weights, constant = Algorithme_Gradient()

# Prédictions sur les données de test
Cpred_test = [1 if Sigmoid(H(weights, constant, D_test[i:i+1, :])) > 0.5 else 0 for i in range(len(D_test))]

# Évaluation des performances
GoodPred = 0
VraiPositifs = 0
FauxPositifs = 0
VraiNegatifs = 0
FauxNegatifs = 0
NumberPositif = 0

for index in range(len(C_test)):
    if (C_test[index] == 1): 
        NumberPositif += 1
    if (C_test[index] == Cpred_test[index]):
        GoodPred += 1
    if (C_test[index] == 1):
        if (Cpred_test[index] == 1):
            VraiPositifs += 1
        else:
            FauxNegatifs += 1
    if (C_test[index] == 0):
        if (Cpred_test[index] == 1):
            FauxPositifs += 1
        else:
            VraiNegatifs += 1

print("Taux de bonnes classifications : ", 100.0 * GoodPred / len(C_test), "%")
print("Quand il fallait prédire 1...")
print(100.0 * VraiPositifs / NumberPositif, "% du temps le prédicteur prédit 1")
print(100.0 * FauxNegatifs / NumberPositif, "% du temps le prédicteur prédit 0")
print("Quand il fallait prédire 0...")
print(100.0 * VraiNegatifs / (len(C_test) - NumberPositif), "% du temps le prédicteur prédit 0")
print(100.0 * FauxPositifs / (len(C_test) - NumberPositif), "% du temps le prédicteur prédit 1")

"""