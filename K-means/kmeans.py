# importer les librairies
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Chargement des données :
df = pd.read_csv("incomeDataSet.csv", sep=";")
print(df.head())


# affichage des données du dataset : on cherche à prédire le salaire en fonction de l'âge
plt.scatter(df['Age'], df['Revenu'])
# plt.show()

km = KMeans(n_clusters=3)
print(km)

Y_pred = km.fit_predict(df[['Age', 'Revenu']])
print(Y_pred)

df['cluster'] = Y_pred
df.head()

df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

plt.scatter(df1.Age, df1['Revenu'], color='green')
plt.scatter(df2.Age, df2['Revenu'], color='red')
plt.scatter(df3.Age, df3['Revenu'], color='black')
plt.xlabel('Age')
plt.ylabel('Revenu')
plt.legend()
plt.show()

# scaler = MinMaxScaler()
# scaler.fit(df[['Revenu']])
# df['Revenu'] = scaler.transform(df['Revenu'])
#
# scaler.fit(df.Age)
# df.Age = scaler.transform(df.Age)
# print(df)

km= KMeans(n_clusters =3)
Y_pred = km.fit_predict(df[['Age', 'Revenu']])
print(Y_pred)

df['cluster'] = Y_pred
df.drop('cluster',axis='columns',inplace=True)
print(df)


df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

plt.scatter(df1.Age, df1['Revenu'], color='green')
plt.scatter(df2.Age, df2['Revenu'], color='red')
plt.scatter(df3.Age, df3['Revenu'], color='black')
plt.scatter()
plt.xlabel('Age')
plt.ylabel('Revenu')
plt.legend()
plt.show()

print(km.clustercenters)