import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

def train_model():
    df = pd.read_csv("data/processed/processed_data.csv")
    X = df[["Humidity", "Wind Speed"]]  
    y = df["Temperature"]
    model = LinearRegression()
    model.fit(X, y)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train_model()
