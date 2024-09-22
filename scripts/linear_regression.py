import json, dotenv, os

def get_old_thetas() -> tuple[float, float]:
    dotenv.load_dotenv()
    env = os.getenv("PWD")