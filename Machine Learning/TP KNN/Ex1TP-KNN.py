import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

#Charger le dataset Iris 
iris = load_iris()
#la taille, les 5 premières lignes, et la distribution des classes:
df=pd.DataFrame(iris.data, columns=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"])
df["species"]=iris.target #étiquettes
df["species"]=df["species"].map({i:name for i, name in enumerate(iris.target_names)})
print("Taille du dataset :", df.shape)
print("Premières lignes :",df.head())#affiche par défaut 5 lignes
print("Distribution des classes :",df["species"].value_counts())
#2) Visualisation
species_colors={"setosa":"red","versicolor":"green","virginica":"blue"}

plt.figure(figsize=(8,6))
for species, color in species_colors.items():#permet de parcourir à la fois l’espèce et sa couleur associée
    subset=df[df["species"]== species] #crée un filtre booléen (True/False)
    plt.scatter(subset["PetalLengthCm"],subset["PetalWidthCm"],label=species, c=color)

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("Petal length vs Petal width 'colorié par éspèce'")
plt.legend()
plt.show()

#Implémentation de KNN

X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("\nPrédictions pour le test :\n", y_pred[:10]) #sélectionne les 10 premières valeurs de y_pred

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=iris.target_names)
print("\nMatrice de confusion :\n", cm)

# Précision
acc = accuracy_score(y_test, y_pred)
print("\nPrécision (Accuracy) avec k=3 :", acc)

accuracies = []
k_values = [1, 3, 5, 7, 9]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc) #ajoute la précision pour ce k dans la liste accuracies.
    print(f"k={k}, Accuracy={acc}")

# Courbe précision vs k
plt.figure(figsize=(8,6))
plt.plot(k_values, accuracies) #(k_values)=axe des x, (accuracies)=axe des y
plt.xlabel("Valeur de k")
plt.ylabel("Accuracy")
plt.title("Précision en fonction de k (KNN sur Iris)")
plt.grid(True)
plt.show()
