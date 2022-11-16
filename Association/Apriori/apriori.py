# importation des bibliothèques utilisées
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns

from mlxtend.frequent_patterns import association_rules, apriori

#lecture du fichier CSV
df = pd.read_csv("bread basket.csv")

#affichage des 5 premières transactions
#print(df.head())

#affichage des informations principales du fichier CSV générées grâce à la fonction describe()
# (le nombre de lignes, le minimum, les quartiles et le maximum)
print("Le document traité comprend un total de transactions de :\n")
print(df.describe().loc[['count']])

print("\nLe nombre de lignes, le minimum, les quartiles et le maximum: \n")
print(df.describe())

#affichage du nom des colonnes
print(df.info())


# Traitement des données avec de la data visualization
# pour afficher les 20 items les plus fréquents
plt.figure(figsize=(15,5))
sns.barplot(x = df.Item.value_counts().head(20).index, y = df.Item.value_counts().head(20).values, palette = 'gnuplot')
plt.xlabel('Items', size = 15)
plt.xticks(rotation=45)
plt.ylabel('Nombre d\'items', size = 15)
plt.title('TOP 20 des items les plus fréquents dans l\'ordre décroissant', color = 'green', size = 20)

#on sauvegarde le graphique dans le dossier
plt.savefig('graphique.png')
# affichaqe du graphique
print(plt.show())

# on regroupe les transactions en ajoutant un compteur
transactions_str = df.groupby(['Transaction', 'Item'])['Item'].count().reset_index(name ='Compteur')
# affichage de toutes les transactions
# avec le numéro de la transaction et le nom du produit
# avec l'ajout d'une colonne compteur
#print(transactions_str)

# création d'une matrice mxn où m=transaction et n=articles
# et chaque ligne représente si l'article était dans la transaction ou non
panier = transactions_str.pivot_table(index='Transaction', columns='Item', values='Compteur', aggfunc='sum').fillna(0)


# création d'une fonction encode
# qui va retourner 1 si le produit est dans la transaction
# 0 si il n'est dans aucune transaction
def encode(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1


# Application de cette fonction sur les données
panier_sets = panier.applymap(encode)
print(panier_sets.head())

supportMinimum = int(input("Veuillez inscrire le support minimum en pourcentage que vous souhaitez tester :"))
supportMinimum = supportMinimum/100

# Utilisation de l'algorithme Apriori
# Recherche des items fréquents avec le support minimum entré
frequent_items = apriori(panier_sets.astype('bool'), min_support = supportMinimum,use_colnames = True)
frequent_items = frequent_items.sort_values('support', ascending=False)
print("\nAffichage des items fréquents \n",
      frequent_items)


# création de règles d'association en fonction des itemset fréquents générées
rules = association_rules(frequent_items, metric = "lift", min_threshold = 1)
rules.sort_values('confidence', ascending = False, inplace = True)


# trie des données de la confidence la plus grande à la plus petite
rules = rules.sort_values('confidence', ascending=False)
#print(rules.sort_values('confidence', ascending=False))
# sauvegarde du document dans le dossier nommé "résultats.csv"
export_csv = rules.to_csv('../Apriori/resutats.csv', index=None, header=True,encoding='utf-8')


#affichage des résultats essentiels
print("\nAffichage des résultats de la confidence la plus grande à la plus petite \n"
      "Uniquement les antecédents, les conséquences, la confidence et le support"
      "Pour plus de détails veuillez consulter le CSV généré dans ce dossier \n")

resultat = pd.DataFrame(rules, columns = ["antecedents", "consequents","confidence", "support"])
print(resultat)


