# ==============================
# Exercice 3 - KNN (Diabetes Dataset)
# ==============================

# 1. Importer les bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score

# 2. Charger le dataset
df = pd.read_csv("diabetes.csv")
print("Nombre d'enregistrements :", len(df))
print("\nAperçu des 10 premières lignes :\n", df.head(10))

# 3. Remplacer les valeurs nulles (0) par la moyenne pour certaines colonnes
cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_with_zeros:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].mean(), inplace=True)

print("\nDonnées après traitement des valeurs nulles :\n", df.head(10))

# 4. Définir les features (X) et la cible (y)
X = df[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]
y = df["Outcome"]

# 5. Diviser en apprentissage (80%) et test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTaille du dataset - Train :", len(X_train), " Test :", len(X_test))

# 6. Mise à l’échelle des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nExemple de données après scaling :\n", X_train_scaled[:5])

# 7. Définir le classifieur KNN (k=5, distance euclidienne)
knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
knn.fit(X_train_scaled, y_train)

# 8. Prédictions
y_pred = knn.predict(X_test_scaled)

# 9. Évaluation du modèle
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Matrice de confusion - KNN (Diabetes)")
plt.show()

print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

# 10. Test avec plusieurs valeurs de k
scores = []
for k in [1, 3, 5, 7, 9, 11]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    y_pred_k = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred_k)
    scores.append((k, acc))

# Tracer courbe accuracy vs k
k_vals, acc_vals = zip(*scores)
plt.plot(k_vals, acc_vals, marker="o")
plt.xlabel("Valeur de k")
plt.ylabel("Accuracy")
plt.title("Variation de la précision en fonction de k")
plt.show()
