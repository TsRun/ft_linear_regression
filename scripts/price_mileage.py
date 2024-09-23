import json
import os
from dotenv import load_dotenv

def getValue(mileage: int, T0: float, T1: float) -> float:
    return T0 + mileage * T1

def main():
    load_dotenv()
    env = os.getenv("PWD")
    try:
        with open(f'{env}/data/gradient.json') as f:
            data = json.load(f)  # Utiliser json.load() pour charger le contenu du fichier
            T0, T1 = map(float, (data['T0'], data['T1']))  # Convertir en float
    except Exception as e:
        T0 = T1 = 0.0  # Initialiser en float
    try:
        while True:
            mileage = int(input("Mileage : "))
            print(getValue(mileage, T0, T1))
    except KeyboardInterrupt:
        print("\nGoodbye !")

if __name__ == "__main__":
    main()
