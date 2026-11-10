import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Données
data = pd.DataFrame({
    "Surface": [50, 65, 80, 120, 100, 90, 150, 70],
    "Chambres": [2, 3, 3, 4, 3, 2, 5, 2],
    "Prix": [150, 200, 220, 330, 280, 250, 400, 210]
})

# Visualisation
plt.scatter(data["Surface"], data["Prix"], color='blue', label="Surface vs Prix")
plt.xlabel("Surface (m²)")
plt.ylabel("Prix (x 1000 DH)")
plt.title("Relation Surface / Prix")
plt.show()

plt.scatter(data["Chambres"], data["Prix"], color='green', label="Chambres vs Prix")
plt.xlabel("Nombre de chambres")
plt.ylabel("Prix (x 1000 DH)")
plt.title("Relation Chambres / Prix")
plt.show()

# Régression linéaire multiple
X = data[["Surface", "Chambres"]]
y = data["Prix"]

model = LinearRegression()
model.fit(X, y)

print("Coefficients :", model.coef_)
print("Intercept :", model.intercept_)

# Prédiction pour 110 m² et 4 chambres
pred = model.predict([[110, 4]])
print(f"Prix prédit : {pred[0]:.2f} (x 1000 DH)")

# Évaluation R²
y_pred = model.predict(X)
print("R² :", r2_score(y, y_pred))

# Visualisation comparaison Réel vs Prédit
plt.scatter(y, y_pred)
plt.plot([min(y), max(y)], [min(y), max(y)], color="red")
plt.xlabel("Prix réel")
plt.ylabel("Prix prédit")
plt.title("Comparaison : Prix réels vs prédits")
plt.show()


