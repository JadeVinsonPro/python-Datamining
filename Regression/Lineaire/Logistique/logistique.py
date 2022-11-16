# importation des librairies
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mlxtend.evaluate import accuracy_score

matplotlib.use('TkAgg')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# importer le Dataset
dataset = pd.read_csv('diabetes-1.csv')

# visualistaion des données
print(dataset.head())

# afficher le dataset (Age-Outcome)
plt.scatter(dataset.Age, dataset.Outcome)
# sauvegarde du graphique dans le dossier
plt.savefig('graphiqueLogistique.png')
# afficher nuage de points
plt.show()

# Supprimer une colonne du dataset
# dataset.drop(['????'], axis='columns', inplace=True)

# Définir notre variable dépendante y et nos variables indépendantes x
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values

# Division du Dataset : en données d'entraînement et tests
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Construction du modèle avec les données d'entraînements
classifier = LogisticRegression(solver='liblinear', random_state=0)
print(classifier.fit(X_train, Y_train))

# Prédiction : en fonction des différents paramètres : âge, grossesses ... déterminez si elle est diabétique ?
Y_pred = classifier.predict(X_test)
print(classifier.score(X_test, Y_test))
# Exemple test : femme âgée de 76 ans et tombée enceinte 3 fois : on essaye de déterminer si oui ou non elle est
# diabètique
#Retourne 1 si oui, sinon 0
print(classifier.predict([[3,110,65,35,0,34,0.125,76]]))

#calculer l'accuracy du modele en comparant les variables y de test et d'entrainement
print("Accuracy : ")
print(accuracy_score(Y_test,Y_pred))