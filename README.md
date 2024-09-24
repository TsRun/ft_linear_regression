# ft_linear_regression

## Introduction
Ce projet vous introduit aux concepts fondamentaux de l'apprentissage automatique en implémentant un algorithme de régression linéaire. L'objectif est de prédire le prix d'une voiture en fonction de son kilométrage à l'aide d'une fonction linéaire entraînée par un algorithme de descente de gradient.

## Table des matières
- [Installation](#installation)
- [Structure du projet](#structure-du-projet)
- [Utilisation](#utilisation)
- [Contributions](#contributions)

## Installation

Pour installer ce projet, suivez les étapes ci-dessous :

1. **Clonez le dépôt :**
```bash
git clone https://github.com/TsRun/ft_linear_regression.git
cd ft_linear_regression
```
 
2. **Installez les dépendances :** 

Assurez-vous d'avoir Python 3 et pip installés. Ensuite, exécutez :

```bash
pip install -r requirement.txt
```

## Structure du projet 


```bash
ft_linear_regression/
├── data/
│   ├── data.csv          # Fichier contenant les données de kilométrage et de prix
│   └── gradient.json     # Fichier pour stocker les paramètres theta
├── requirement.txt       # Liste des dépendances
└── scripts/
    ├── linear_regression.py  # Script principal pour la régression linéaire
    ├── price_mileage.py      # Script pour prédire le prix d'une voiture
    └── test_gradient.py       # Tests pour valider l'algorithme
```

## Utilisation 
 
1. **Entraînement du modèle :** 
Exécutez le script suivant pour entraîner votre modèle avec le jeu de données :

```bash
python scripts/linear_regression.py
```
 
2. **Prédiction du prix :** 
Une fois le modèle entraîné, vous pouvez prédire le prix d'une voiture en fonction de son kilométrage :

```bash
python scripts/price_mileage.py
```

## Contributions 

Les contributions sont les bienvenues ! N'hésitez pas à soumettre des pull requests ou à signaler des problèmes.


```vbnet
N'hésite pas à personnaliser les sections selon tes préférences ou à ajouter des détails supplémentaires si nécessaire !
```
