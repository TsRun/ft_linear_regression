import json, dotenv, os
from pandas import read_csv, DataFrame


def get_old_thetas(gradiant_path) -> tuple[float, float]:
    with open(gradiant_path) as f:
        data = json.load(f)
        T0, T1 = map(float, (data['T0'], data['T1']))
    return T0, T1

dotenv.load_dotenv()
env = os.getenv("PWD")
if not env:
    raise Exception("PWD is not defined in your environment variables")
gradiant_path = os.path.join(env, 'data', 'gradiant.json')
T0, T1 = get_old_thetas(gradiant_path=gradiant_path)
    
def get_data(data_path) -> DataFrame:
    data = read_csv(data_path)
    return data

def estimatePrice(mileage : int, T0, T1):
    return T0 + T1 * mileage

def linear_regression(data : DataFrame, T0 : float, T1 : float, learning_rate : float = 0.1) -> tuple[float, float]:
    tmpT0 = learning_rate * (sum([1]))

def main():
    if not env:
        raise Exception("PWD is not defined in your environment variables")
    data_path = os.path.join(env, 'data', 'data.csv')
    data = get_data(data_path=data_path)
    tmpT0, tmpT1 = linear_regression(data, T0, T1)

if __name__ == "__main__":
    main()