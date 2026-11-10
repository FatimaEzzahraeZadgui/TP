import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Données 
X = np.array([[6], [8], [10], [14], [18]])
y = np.array([7, 9, 13, 17.5, 18])

#  Visualisation des données 
plt.scatter(X, y, color='blue', label='Données')
plt.xlabel("Taille de la pizza (pouces)")
plt.ylabel("Prix (DH)")
plt.title("Prix des pizzas en fonction de leur taille")
plt.legend()
plt.show()


#Régression linéaire 
model = LinearRegression()
model.fit(X, y)

print("Coefficient (a) :", model.coef_[0]) #combien le prix augmente
print("Intercept (b) :", model.intercept_) #le prix estimé quand la taille est 0

# Prédiction pour 12 pouces
prix_12 = model.predict([[12]])
print(f"Prix prédit pour une pizza de 12 pouces : {prix_12[0]:.2f} DH")

# Visualisation de la droite de régression
plt.scatter(X, y, color='blue', label='Données')#nuage de points
plt.plot(X, model.predict(X), color='red', label='Régression linéaire')
plt.xlabel("Taille de la pizza")
plt.ylabel("Prix")
plt.title("Régression linéaire : Prix vs Taille")
plt.legend()
plt.show()


#Évaluation du modèle
y_pred = model.predict(X) 
score = r2_score(y, y_pred)
print("Coefficient de détermination R² :", score)
