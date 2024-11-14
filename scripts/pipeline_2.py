import datetime
import os
from typing import Dict

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from mlflow.models import EvaluationResult, MetricThreshold, infer_signature
from prefect import flow, get_run_logger, task
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             roc_curve)

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Define the baseline model URI (replace manually as needed)
logged_model = 'runs:/836b6df347c241e2803d8b8b072fa2af/artifacts'
baseline_model_uri = logged_model
#baseline_model_uri = None


# Define accuracy threshold
thresholds: Dict[str, MetricThreshold] = {
    "accuracy_score": MetricThreshold(
        threshold=0.85,  
        greater_is_better=True,
    ),
}

experiment_tags: Dict[str, str] = {
    "project_name": "heart-disease-forecasting",
    "store_dept": "health",
    "team": "stores-ml",
    "project_quarter": "Q4-2024",
    "mlflow.note.content": "This is the Heart disease prediction project.",
}

@task(retries=20, retry_delay_seconds=10)
def train_and_log_model() -> None:
    """Train a logistic regression model, log metrics, and artifacts to MLflow."""
    experiment_name = "Health Models Experiment 1"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(name=experiment_name, tags=experiment_tags)
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

    # Create artifacts directory
    artifacts_path = "artifacts"
    os.makedirs(artifacts_path, exist_ok=True)

    # Save the ROC curve as an artifact
    roc_path = os.path.join(artifacts_path, "roc_curve.png")
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(roc_path)
    plt.close()

    # Save the confusion matrix as an artifact
    cm_path = os.path.join(artifacts_path, "confusion_matrix.png")
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
    )
    plt.title("Confusion Matrix")
    plt.savefig(cm_path)
    plt.close()

    # Save metrics to a CSV file
    metrics_df = pd.DataFrame({"accuracy": [accuracy], "f1_score": [f1]})
    metrics_csv_path = os.path.join(artifacts_path, "metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)

    # Define the model signature
    signature = infer_signature(X_train, y_train)

    # Log metrics and artifacts in MLflow
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(metrics_csv_path)

        # Log the model to the Model Registry
        model_name = "LogisticRegression_Heart_Model"
        mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)

        # Evaluate and compare with baseline model
        eval_data = X_test.copy()
        eval_data["target"] = y_test
        evaluate_and_compare(run, artifacts_path, model, eval_data, signature)

    logger = get_run_logger()
    logger.info(f"Logistic regression model trained successfully: Accuracy={accuracy}, F1 Score={f1}")

    if accuracy < thresholds["accuracy_score"].threshold:
        raise ValueError(f"Accuracy {accuracy:.2f} below threshold {thresholds['accuracy_score'].threshold}, retry triggered.")

@task
def evaluate_and_compare(run, artifact_path: str, candidate_model, eval_data: pd.DataFrame, signature) -> None:
    model_name = "LogisticRegression_Heart_Model"
    model_uri = mlflow.sklearn.log_model(
        sk_model=candidate_model,
        artifact_path=artifact_path,
        signature=signature
    ).model_uri

    mlflow.register_model(model_uri, model_name)

    if baseline_model_uri:
        evaluation_result: EvaluationResult = mlflow.evaluate(
            model=run.info.artifact_uri + "/artifacts",
            data=eval_data,
            targets="target",
            model_type="classifier",
            validation_thresholds=thresholds,
            baseline_model=baseline_model_uri
        )
        print("Evaluation Result:", evaluation_result.metrics)

@flow(name="data-quality-training-pipeline")
def main_flow() -> None:
    train_and_log_model()

if __name__ == "__main__":
    main_flow()
