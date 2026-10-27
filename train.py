
from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score

import mlflow
import mlflow.sklearn

# Configure MLflow tracking URI and experiment name
# ml flow server stores experiment metadata and model records
mlflow.set_tracking_uri("http://127.0.0.1:6001") 

# group related training runs under a common experiment name
mlflow.set_experiment("iris_classification")




iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.8, random_state=42
)

n_estimators = 100
random_state = 42
# Model settings we want to trackk in MLflow
model_parameters={
    "model_type":"RandomForestClassifier",
    "n_estimators":n_estimators,
    "random_state":random_state,
    "test_size":0.2
}
# Start a tracked training run
with mlflow.start_run() as run:
    # Record the training configuration before model fitting
    mlflow.log_params(model_parameters)
    # Build classifier using the selected paramters
    model=RandomForestClassifier(
        n_estimators=n_estimators, 
        random_state=random_state)
    # Fit the model on training data
    model.fit(X_train, y_train)

    # generate predictions using test data
    y_pred=model.predict(X_test)

    # Calculate metrics that help us evaluate the model performance
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred,average="weighted")
    recall=recall_score(y_test,y_pred,average="macro",zero_division=0)

    # Log the evaluated results so that different run can be compared
    mlflow.log_metrics({
        "test_accuracy":accuracy,
        "test_precision":precision,
        "test_recall":recall
    })      
    print("MLFlow run ID:",run.info.run_id)
    print("Test Accuracy:",accuracy)
    print("Test Precision:",precision)
    print("Test Recall:",recall)


    # Save the trained model as reusable artifact for later use in the API
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "iris_model.joblib"
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")

    # Log the trained model to MLflow
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="iris_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    # Register the logged model in the MLflow Model Registry
    registered_model = mlflow.register_model(
        model_uri=model_info.model_uri,
        name="iris_classifier"
    )

    print("Registered Model : iris_classifier")
    print("Model Version:", registered_model.version)