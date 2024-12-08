import pandas as pd

def preprocess_data():
    df = pd.read_csv("data/raw/raw_data.csv")
    df.dropna(inplace=True) 
    df["Temperature"] = (df["Temperature"] - df["Temperature"].mean()) / df["Temperature"].std()  # Normalize
    df["Wind Speed"] = (df["Wind Speed"] - df["Wind Speed"].mean()) / df["Wind Speed"].std()  # Normalize
    df.to_csv("data/processed/processed_data.csv", index=False)

if __name__ == "__main__":
    preprocess_data()
