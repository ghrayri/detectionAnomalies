# Importation des bibliothèques
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

## importation des donées
data = pd.read_csv("C:/Users/votre/chemin/creditcard.csv")
print(data)



# Vérification des valeurs manquantes
print(data.isnull().sum())

#diviser les données 
X = data.iloc[:, :-1] 
y = data.iloc[:, -1]  

# Normaliser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  

# Division en ensemble d'entraînement et de test
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

## Étape 3 : Modèle Isolation Forest
# Initialisation du modèle
iso_forest = IsolationForest(contamination=0.02, random_state=42)

# Entraînement 
iso_forest.fit(X_train)

# Prédictions 
train_predictions = iso_forest.predict(X_train)
test_predictions = iso_forest.predict(X_test)

# Conversion des prédictions : -1 -> 1 (Anomalie), 1 -> 0 (Normal)
train_predictions = np.where(train_predictions == -1, 1, 0)
test_predictions = np.where(test_predictions == -1, 1, 0)

## Étape 4 : Analyse des anomalies
# Scores d'anomalie
train_scores = iso_forest.decision_function(X_train)
test_scores = iso_forest.decision_function(X_test)

# Nombre d'anomalies détectées
print(f"Nombre d'anomalies détectées dans l'ensemble d'entraînement : {sum(train_predictions)}")
print(f"Nombre d'anomalies détectées dans l'ensemble de test : {sum(test_predictions)}")

## Étape 5 : Visualisation des résultats
# Distribution des scores d'anomalie pour l'ensemble de test
plt.figure(figsize=(10, 6))
sns.histplot(test_scores, kde=True, bins=50)
plt.title("Distribution des scores d'anomalie (Test)")
plt.xlabel("Score d'anomalie")
plt.ylabel("Fréquence")
plt.show()

# Distribution des prédictions dans l'ensemble de test
plt.figure(figsize=(10, 6))
sns.countplot(x=test_predictions)
plt.title("Distribution des prédictions (0: normal, 1: anomalie)")
plt.xlabel("Prédictions")
plt.ylabel("Nombre de cas")
plt.show()
