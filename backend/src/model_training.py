import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
import mlflow
import mlflow.sklearn

def train_model():
    # Load data
    df = pd.read_csv("data/processed/processed_data.csv")
    X = df[["Humidity", "Wind Speed"]]  
    y = df["Temperature"]

    # Start MLflow experiment
    mlflow.set_experiment("Linear Regression Temperature Prediction")

    with mlflow.start_run():
        # Define model and parameters
        model = LinearRegression()
        mlflow.log_param("model_type", "LinearRegression")
        
        # Train model
        model.fit(X, y)

        # Make predictions for training set
        y_pred = model.predict(X)

        # Calculate and log metrics
        mse = mean_squared_error(y, y_pred)
        mlflow.log_metric("mse", mse)

        # Save the trained model
        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)

        # Log the model using MLflow
        mlflow.sklearn.log_model(model, "model")

        print(f"Model trained and logged with MSE: {mse}")

if __name__ == "__main__":
    train_model()