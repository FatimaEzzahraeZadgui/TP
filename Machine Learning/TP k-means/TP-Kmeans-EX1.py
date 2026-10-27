import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = np.array([
    [1, 2],  #A
    [1, 4],  #B
    [1, 0],  #C
    [10, 2], #D
    [10, 4], #E
    [10, 0]  #F
])


kmeans = KMeans(n_clusters=2, init='k-means++', random_state=0) #contraire à inicialiser les centroide avec random (init='random')
y_kmeans = kmeans.fit_predict(X)


plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')

#Visualiser les centroïdes
plt.scatter(
    kmeans.cluster_centers_[:, 0], #abscisses
    kmeans.cluster_centers_[:, 1], #ordonées
    s=300,
    c='yellow',
    label='Centroïdes'
)
plt.title('Résultat du K-Means (k=2)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

