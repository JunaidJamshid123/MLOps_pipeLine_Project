import os
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# Configure MLflow to store logs locally
mlflow.set_tracking_uri("file:///" + os.path.abspath("mlruns").replace("\\", "/"))

def train_model():
    # Load dataset
    df = pd.read_csv("data/processed/processed_data.csv")
    X = df[["Humidity", "Wind Speed"]]
    y = df["Temperature"]

    # Set experiment name
    mlflow.set_experiment("Temperature Prediction Experiment")

    with mlflow.start_run():
        # Train model
        model = LinearRegression()
        model.fit(X, y)

        # Predictions and metrics
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)

        # Log parameters, metrics, and model
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("features", ["Humidity", "Wind Speed"])
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "model")

        print(f"Model trained with MSE: {mse}")
        # Save the trained model
        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)

        # Register the model locally (simulated without centralized Model Registry)
        model_uri = mlflow.get_artifact_uri("model")
        print(f"Model saved to: {model_uri}")

if __name__ == "__main__":
    train_model()