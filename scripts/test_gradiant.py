import json
import dotenv
import os
import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
import numpy as np

# Fonction de calcul du coût
def compute_cost(T0, T1, X, Y):
    m = len(Y)
    predictions = T0 + T1 * X
    error = predictions - Y
    cost = (1 / (2 * m)) * np.sum(error ** 2)
    return cost

# Fonction de calcul des gradients analytiques
def compute_analytical_gradients(T0, T1, X, Y):
    m = len(Y)
    predictions = T0 + T1 * X
    error = predictions - Y
    grad_T0 = (1 / m) * np.sum(error)
    grad_T1 = (1 / m) * np.sum(error * X)
    return grad_T0, grad_T1

# Fonction de calcul des gradients numériques
def compute_numerical_gradient(T0, T1, X, Y, epsilon=1e-4):
    # Calcul du gradient pour T0
    cost_plus = compute_cost(T0 + epsilon, T1, X, Y)
    cost_minus = compute_cost(T0 - epsilon, T1, X, Y)
    grad_T0_num = (cost_plus - cost_minus) / (2 * epsilon)

    # Calcul du gradient pour T1
    cost_plus = compute_cost(T0, T1 + epsilon, X, Y)
    cost_minus = compute_cost(T0, T1 - epsilon, X, Y)
    grad_T1_num = (cost_plus - cost_minus) / (2 * epsilon)

    return grad_T0_num, grad_T1_num

# Fonction de vérification des gradients
def gradient_check(T0, T1, X, Y, epsilon=1e-4):
    # Calcul des gradients analytiques
    grad_T0_analytical, grad_T1_analytical = compute_analytical_gradients(T0, T1, X, Y)
    
    # Calcul des gradients numériques
    grad_T0_numerical, grad_T1_numerical = compute_numerical_gradient(T0, T1, X, Y, epsilon)
    
    # Comparaison des deux gradients
    diff_T0 = np.abs(grad_T0_analytical - grad_T0_numerical) / (np.abs(grad_T0_analytical) + np.abs(grad_T0_numerical) + epsilon)
    diff_T1 = np.abs(grad_T1_analytical - grad_T1_numerical) / (np.abs(grad_T1_analytical) + np.abs(grad_T1_numerical) + epsilon)
    
    # Affichage des résultats
    print(f"Gradient T0 - Analytique: {grad_T0_analytical}, Numérique: {grad_T0_numerical}, Différence: {diff_T0}")
    print(f"Gradient T1 - Analytique: {grad_T1_analytical}, Numérique: {grad_T1_numerical}, Différence: {diff_T1}")
    
    return diff_T0, diff_T1

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

def main():
    dotenv.load_dotenv()
    env = os.getcwd()
    data_path = os.path.join(env, 'data', 'data.csv')
    
    data = get_data(data_path=data_path)
    
    X = data['km'].values
    Y = data['price'].values
    
    # Normaliser les données
    X_norm, mean_X, std_X = normalize_feature(X)
    Y_norm, mean_Y, std_Y = normalize_feature(Y)
    
    # Valeurs initiales de T0 et T1
    T0 = 0.0
    T1 = 0.0
    
    # Exécution du gradient check
    diff_T0, diff_T1 = gradient_check(T0, T1, X_norm, Y_norm)

    # Vérifier la précision
    if diff_T0 < 1e-7 and diff_T1 < 1e-7:
        print("Les gradients sont corrects.")
    else:
        print("Les gradients semblent incorrects. Veuillez vérifier vos calculs.")

if __name__ == "__main__":
    main()
