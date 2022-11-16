# importer les librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from sklearn.cluster import KMeans
from sklearn import datasets

# chargement de base de données iris
iris = datasets.load_iris()

# affichage des données du dataset
print(iris.target)
print(iris.target_names)
print(iris.feature_names)
print(iris.data)
# stocker les données en tant que DataFrame Pandas
x = pd.DataFrame(iris.data, columns=['Sepal_Length', 'Sepal_width', 'Petal_Length', 'Petal_width'])

# définir les noms de colonnes
# print(x.columns['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
# y=pd.DataFrame(iris.target)
# print(y.columns['classe'])

# Répartition du DataSet dans un scatter plot 2D
plt.scatter(x.Petal_Length, x.Petal_width)
plt.show()
# Visualiser les classes de notre dataset
colorL = np.array(['green', 'red', 'blue'])
plt.scatter(x.Petal_Length, x.Petal_width, c=colorL[iris.target], s=20)
plt.show()
# Utiliser la méthode Elbow pour trouver le nombre optimal de clusters

x1 = x.iloc[:, [0, 1, 2, 3]]
SSE = []
Krange = np.arange(1, 11)
for k in Krange:
    KM = KMeans(n_clusters=k)
    KM.fit(x)
    SSE.append(KM.inertia_)
# squarred errors
print(SSE)

# Afficher la courbe de la méthode elbow

plt.plot(Krange, SSE)
plt.title('La méthode Elbow')
plt.xlabel('Nombre de clusters')
plt.ylabel('inertia')
plt.show()

# Clusters K-Means
model = KMeans(n_clusters=3)
print(model.fit(x))

print(model.labels_)

# Visualiser les classes prédites par le modèle
colorL = np.array(['green', 'red', 'blue'])
plt.scatter(x.Petal_Length, x.Petal_width, c=colorL[model.labels_], s=20)

# Visualiser les classes originales et prédites par le modèle
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
fig.suptitle('classe originales et classes trouvées par le modèle')
ax1.scatter(x.Petal_Length, x.Petal_width, c=colorL[model.labels_], s=20)
ax2.scatter(x.Petal_Length, x.Petal_width, c=colorL[iris.target], s=20)

# Matrice de confusion
from sklearn.metrics import confusion_matrix

print(confusion_matrix((iris.target), model.labels_))