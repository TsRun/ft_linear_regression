import json
import dotenv
import os
import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
import numpy as np

def save_thetas(T0, T1, gradient_path):
    with open(gradient_path, 'w') as f:
        json.dump({'T0': T0, 'T1': T1}, f, indent=4)

def get_old_thetas(gradient_path, mean_X, std_X, mean_Y, std_Y) -> tuple[float, float]:
    try:
        with open(gradient_path, 'r') as f:
            data = json.load(f)
            T0, T1 = map(float, (data['T0'], data['T1']))
            # Renormalisation des thetas
            T1_renorm = T1 * (std_X / std_Y)
            T0_renorm = (T0 - mean_Y + T1 * mean_X) / std_Y
        return T0_renorm, T1_renorm
    except FileNotFoundError:
        print(f"Fichier {gradient_path} non trouvé. Initialisation des thetas à 0.")
        return 0.0, 0.0

def get_data(data_path) -> DataFrame:
    data = read_csv(data_path)
    if 'km' not in data.columns or 'price' not in data.columns:
        raise ValueError("Le fichier CSV doit contenir les colonnes 'km' et 'price'.")
    return data

def normalize_feature(X):
    """Standardize the feature X."""
    mean = np.mean(X)
    std = np.std(X)
    return (X - mean) / std, mean, std

def denormalize_thetas(T0, T1, mean_X, std_X, mean_Y, std_Y):
    """Dénormalise T0 et T1 pour revenir à l'échelle d'origine."""
    T1_denorm = T1 * (std_Y / std_X)
    T0_denorm = (T0 * std_Y) + mean_Y - (T1_denorm * mean_X)
    return T0_denorm, T1_denorm

def linear_regression(data: DataFrame, T0: float, T1: float, learning_rate: float, iterations: int) -> tuple[list, list, list, float, float, float, float]:
    X = data['km'].values
    Y = data['price'].values
    m = len(Y)
    
    # Normalisation des caractéristiques et des cibles
    X_norm, mean_X, std_X = normalize_feature(X)
    Y_norm, mean_Y, std_Y = normalize_feature(Y)
    
    history_T0, history_T1 = [], []
    history_cost = []

    for iteration in range(iterations):
        predictions = T0 + T1 * X_norm
        error = predictions - Y_norm
        cost = (1 / (2 * m)) * np.sum(error ** 2)
        
        # Calcul des gradients
        tmpT0 = (1 / m) * np.sum(error)
        tmpT1 = (1 / m) * np.sum(error * X_norm)

        # Mise à jour des paramètres
        T0 -= learning_rate * tmpT0
        T1 -= learning_rate * tmpT1

        history_T0.append(T0)
        history_T1.append(T1)
        history_cost.append(cost)

        if iteration % 100 == 0:  # Afficher tous les 100 itérations
            print(f"Iteration {iteration}: T0 = {T0}, T1 = {T1}, Cost = {cost}")

        # Vérifier si les valeurs sont infinies ou NaN
        if np.isnan(T0) or np.isnan(T1) or np.isnan(cost):
            print("Divergence détectée. Arrêt de l'algorithme.")
            break

    return history_T0, history_T1, history_cost, mean_X, std_X, mean_Y, std_Y

def plot_progression(history_T0, history_T1, history_cost, data, mean_X, std_X, mean_Y, std_Y):
    plt.figure(figsize=(10, 6))
    
    # Tracer les points de données réels
    plt.scatter(data['km'], data['price'], color='blue', label='Données réelles')

    # Tracer la ligne de régression à la dernière étape
    final_T0 = history_T0[-1]
    final_T1 = history_T1[-1]
    X = data['km'].values
    X_norm = (X - mean_X) / std_X
    regression_line = final_T0 + final_T1 * X_norm
    
    # Dénormalisation de la ligne de régression pour l'échelle des prix
    regression_line = regression_line * std_Y + mean_Y
    plt.plot(X, regression_line, color='red', label='Ligne de régression')

    # Titres et labels
    plt.title('Ajustement de la Régression Linéaire (Normalisation complète)')
    plt.xlabel('Kilométrage (km)')
    plt.ylabel('Prix (€)')
    plt.legend()
    plt.show()

    # Tracer la progression des thetas
    plt.figure(figsize=(10, 6))
    plt.plot(history_T0, label='Theta 0 (T0)', color='green')
    plt.plot(history_T1, label='Theta 1 (T1)', color='purple')
    plt.title('Progression des Thetas durant la Descente de Gradient')
    plt.xlabel('Itérations')
    plt.ylabel('Valeur des Thetas')
    plt.legend()
    plt.show()

    # Tracer la progression du coût
    plt.figure(figsize=(10, 6))
    plt.plot(history_cost, label='Coût (Cost)', color='orange')
    plt.title('Progression du Coût durant la Descente de Gradient')
    plt.xlabel('Itérations')
    plt.ylabel('Coût')
    plt.legend()
    plt.show()

def main():
    dotenv.load_dotenv()
    env = os.getcwd()
    gradient_path = os.path.join(env, 'data', 'gradient.json')
    data_path = os.path.join(env, 'data', 'data.csv')
    
    data = get_data(data_path=data_path)
    
    # Normalisation des données
    X = data['km'].values
    Y = data['price'].values
    X_norm, mean_X, std_X = normalize_feature(X)
    Y_norm, mean_Y, std_Y = normalize_feature(Y)
    
    # Récupérer les thetas et les renormaliser
    T0, T1 = get_old_thetas(gradient_path=gradient_path, mean_X=mean_X, std_X=std_X, mean_Y=mean_Y, std_Y=std_Y)
    
    # Ajustez le taux d'apprentissage
    learning_rate = 1
    # Nombre d'itérations
    iterations = 10000
    
    history_T0, history_T1, history_cost, mean_X, std_X, mean_Y, std_Y = linear_regression(
        data, T0, T1, learning_rate=learning_rate, iterations=iterations
    )
    
    if len(history_cost) == iterations:
        T0_denorm, T1_denorm = denormalize_thetas(history_T0[-1], history_T1[-1], mean_X, std_X, mean_Y, std_Y)
        save_thetas(T0_denorm, T1_denorm, gradient_path)

        plot_progression(history_T0, history_T1, history_cost, data, mean_X, std_X, mean_Y, std_Y)

if __name__ == "__main__":
    main()
