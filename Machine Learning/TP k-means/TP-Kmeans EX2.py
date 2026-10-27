import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataset = pd.read_csv('Mall_Customers.csv')

print(dataset.head())
print(dataset.tail())

print(dataset.shape)

#Définir les données d’entrée X
X = dataset.iloc[:, [3, 4]].values #iloc fct de pandas pour integer location(selectionne par indice) .values converti en tableau numpy

#Afficher X
print("les 10 premières lignes :\n",X[:10])  # pour voir les 10 premières lignes

wcss = [] 

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',  # meilleure initialisation
        max_iter=300,      # nb max d’itérations
        n_init=10,         # nb de répétitions avec diff. centres initiaux
        random_state=0
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    print("WCSS",wcss)

#Visualiser la méthode du coude
plt.plot(range(1, 11), wcss)
plt.title('Méthode du coude')
plt.xlabel('Nombre de clusters (K)')
plt.ylabel('WCSS')
plt.show()


#Choisir K optimal (5)
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    max_iter=300,
    n_init=10,
    random_state=0
)
Y_kmeans = kmeans.fit_predict(X)


#Visualiser les clusters
plt.scatter(X[Y_kmeans == 0, 0], X[Y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[Y_kmeans == 1, 0], X[Y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[Y_kmeans == 2, 0], X[Y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[Y_kmeans == 3, 0], X[Y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[Y_kmeans == 4, 0], X[Y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')


#Visualiser les centroïdes
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c='yellow',
    label='Centroids'
)

#Ajouter les titres et légendes
plt.title('Clusters de clients (segmentation K-Means)')
plt.xlabel('Revenu annuel (k$)')
plt.ylabel('Score de dépense (1-100)')
plt.legend()
plt.show()
