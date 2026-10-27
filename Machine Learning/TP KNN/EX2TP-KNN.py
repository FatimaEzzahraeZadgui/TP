import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Charger le dataset CSV (il faut que tu aies le fichier student-mat.csv dans le même dossier)
df = pd.read_csv("student-mat.csv", sep=";")  # le séparateur du fichier est ";"

# Créer la colonne 'result' : Pass si G3 >=10, Fail sinon
def note_result(x):
    if x >= 10:
        return "Pass"
    else:
        return "Fail"

# Appliquer cette fonction à la colonne G3
df['result'] = df['G3'].apply(note_result)


# Sélectionner les variables explicatives
features = ['studytime', 'failures', 'absences', 'G1', 'G2']
X = df[features]
y = df['result']

# Vérifier le DataFrame
print(df.head())
plt.figure(figsize=(8,6))
plt.hist(df['G3'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel("Note G3")
plt.ylabel("Nombre d'étudiants")
plt.title("Distribution des notes finales (G3)")
plt.show()

print("Nombre de Pass vs Fail :\n", df['result'].value_counts())

# Diviser en 70% train / 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Appliquer KNN avec k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Prédiction
y_pred = knn.predict(X_test)


cm = confusion_matrix(y_test, y_pred, labels=['Pass','Fail'])
print("Matrice de confusion :\n", cm)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='Pass')
recall = recall_score(y_test, y_pred, pos_label='Pass')
f1 = f1_score(y_test, y_pred, pos_label='Pass')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

k_values = [1, 3, 5, 7, 11]
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"k={k}, Accuracy={acc:.3f}")

# Tracer la courbe accuracy vs k
plt.figure(figsize=(8,6))
plt.plot(k_values, accuracies)
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Précision du modèle KNN selon k")
plt.grid(True)
plt.show()
