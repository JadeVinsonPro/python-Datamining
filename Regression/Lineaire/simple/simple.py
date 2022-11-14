# importer les librairies
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# importer le DataSet
Data = pd.read_csv('DataSetLoyersMaisons.csv', sep=";")
# exclure les données dont le loyer supérieur à  10000
Data = Data[Data['loyer'] <= 10000]
print(Data.head())

# afficher nuage de points
x = Data['surface']
y = Data['loyer']
plt.scatter(x, y, color='blue')
plt.show()

X = Data.iloc[:,:-1].values
Y = Data.iloc[:,-1].values

print(X.shape)
print(Y.shape)

# Division du Dataset : en données d'entraînement et tests(1/3) et 2/3 à l'apprentissage
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=1.0 / 3)

print(len(X))
# 2/3 de X
print(len(X_train))
#len(X_test)=len(X_train)/len(X)
print(len(X_test))

# Construire le modèle avec les données d'entraînements
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Prédictions : du prix de loyers des maisons qui sont dans la base de test comparé à Y_test
Y_pred = regressor.predict(X_test)
print(Y_pred)

#print(regressor.predict([[125]]))
# Visualiser le nuage de points
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Evolution des loyers par surface')
plt.show()