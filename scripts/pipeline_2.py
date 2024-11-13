import os
import datetime
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from prefect import flow, get_run_logger, task
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, roc_curve
from datetime import timedelta

@task(retries=3, retry_delay_seconds=10)
def train_and_log_model():
    """Train a logistic regression model, log metrics and artifacts to MLflow."""
    mlflow.set_tracking_uri("mlflow_run")  
    experiment_name = "experience_1"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # Load cleaned data
    X_train = pd.read_csv(r"data/X_train_clean.csv")
    X_test = pd.read_csv(r"data/X_test_clean.csv")
    y_train = pd.read_csv(r"data/y_train.csv").squeeze()
    y_test = pd.read_csv(r"data/y_test.csv").squeeze()
    
    # Initialize and train the logistic regression model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predict and calculate metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Calculate ROC curve and confusion matrix
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)
    
    # Create a temporary directory for artifacts
    artifacts_path = "artifacts"
    os.makedirs(artifacts_path, exist_ok=True)
    
    # Save the ROC curve as an artifact
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    roc_path = os.path.join(artifacts_path, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()

    # Save the confusion matrix as an artifact
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
    plt.title("Confusion Matrix")
    cm_path = os.path.join(artifacts_path, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # Save metrics to a CSV file
    metrics_df = pd.DataFrame({"accuracy": [accuracy], "f1_score": [f1]})
    metrics_csv_path = os.path.join(artifacts_path, "metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)

    # Log artifacts to MLflow
    with mlflow.start_run() as run:
        run_id = run.info.run_id  # Capture the run ID
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        # Log artifacts to MLflow
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(metrics_csv_path)
        
        # Log the model to the Model Registry
        model_name = "LogisticRegression_Heart_Model"
        mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)

    logger = get_run_logger()
    logger.info(f"Logistic regression model trained successfully: Accuracy={accuracy}, F1 Score={f1}")

@flow(name="data-quality-training-pipeline")
def main_flow():
    train_and_log_model()

# Exécuter le flow avec un intervalle en utilisant `.serve()`
if __name__ == "__main__":
    #main_flow()
    main_flow.serve(name="5-min-schedule", cron="*/5 * * * *")  # Toutes les 5 minutes
