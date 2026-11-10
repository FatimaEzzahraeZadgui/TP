##### Exercice 3 ####################################
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

# 1) Création du dataset
data = {
    "Self_Study_Daily": [1, 2, 3, 4, 5, 6, 7, 8],
    "Tution_Monthly":   [2, 4, 3, 6, 8, 5, 7, 9],
    "Pass_Or_Fail":     [0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)

# 2) Séparation variables
X = df[["Self_Study_Daily", "Tution_Monthly"]]
y = df["Pass_Or_Fail"]

# 3) Modèle logistique
model = LogisticRegression()
model.fit(X, y)

print("Coefficients :", model.coef_)#B1 et B2
print("Intercept :", model.intercept_)#B0

# 4) Prédictions + Accuracy
y_pred = model.predict(X)
print("Accuracy :", accuracy_score(y, y_pred))

# 5) Affichage de la frontière de décision avec DecisionBoundaryDisplay
disp = DecisionBoundaryDisplay.from_estimator(model, X, response_method="predict", cmap="bwr", alpha=0.3) # ou "predict_proba" pour une probabilité

# Ajout des points du dataset
plt.scatter(X["Self_Study_Daily"], X["Tution_Monthly"], c=y, cmap="bwr")

# Mise en forme
plt.xlabel("Self Study Daily (h/jour)")
plt.ylabel("Tuition Monthly (h/mois)")
plt.title("Frontière de décision (Pass/Fail)")
plt.grid(True)  # quadrillage
plt.show()
