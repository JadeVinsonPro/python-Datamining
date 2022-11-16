import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth


#ajout des données où appliquer l'algorithme, extrait des transactions d'un supermarché
data = [ ['Yougurt', 'Coffee Powder', 'Ghee', 'Sugar'],
         ['Yougurt', 'Ghee', 'Cheese', 'Panner', 'Lassi', 'Sugar', ''],
        ['Lassi', 'Tea Powder'],
          ['Yougurt', 'Cheese', 'Bread', 'Milk', 'Butter', 'Sugar', 'Lassi'],
          ['Ghee', 'Butter', 'Sweet', 'Milk', 'Lassi', 'Panner'],
          ['Butter', 'Lassi', 'Milk', 'Coffee Powder'],
          ['Coffee Powder', 'Bread', 'Lassi', 'Yougurt', 'Panner', 'Sugar'],
          ['Panner', 'Ghee', 'Butter', 'Yougurt'],
          ['Coffee Powder', 'Sweet', 'Milk', 'Tea Powder', 'Lassi', 'Cheese'],
          ['Cheese', 'Yougurt'],
          ['Tea Powder', 'Sugar', 'Panner', 'Cheese', 'Bread'],
          ['Tea Powder', 'Sweet', 'Lassi', 'Butter'],
          ['Butter', 'Milk', 'Sugar', 'Panner'],
          ['Yougurt', 'Coffee Powder', 'Panner', 'Bread', 'Sweet', 'Ghee', 'Cheese', 'Lassi'],
          ['Sweet', 'Tea Powder', 'Milk', 'Lassi', 'Ghee', 'Coffee Powder'],
          ['Panner', 'Sugar', 'Bread', 'Milk', 'Cheese'],
          ['Sugar', 'Bread', 'Coffee Powder', 'Cheese'],
          ['Bread', 'Cheese', 'Yougurt', 'Milk', ''],
         ['Ghee', 'Bread', 'Yougurt', 'Sugar', 'Cheese'],
          ['Panner', 'Yougurt', 'Bread']]

te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
# création d'une matrice mxn où m=transaction et n=articles
# et chaque ligne représente si l'article était dans la transaction ou non
# retourne True si le produit est dans la transaction
# False si il n'est dans aucune transaction
df = pd.DataFrame(te_ary, columns=te.columns_)

print(df)
#affichage des résultats sans le nom des items fréquents

print(fpgrowth(df, min_support=0.2))

#affichage des résultats avec le nom des items fréquents
print(fpgrowth(df, min_support=0.2, use_colnames=True))



