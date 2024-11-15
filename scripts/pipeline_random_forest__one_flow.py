import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from mlflow.models import EvaluationResult, MetricThreshold, infer_signature
from prefect import flow, get_run_logger, task
from preprocessing import DataPreprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")


#  Load the model from the previous experiment run
logged_model = "runs:/09ac50ed8bb54154a919a84de0315ccd/artifacts"

# baseline_model_uri = None
baseline_model_uri = logged_model

# Define criteria for model to be validated against
""" Autre parametre de validation 
    - min_absolute_change=0.05,  # accuracy should be at least 0.05 greater than baseline model accuracy
    - min_relative_change=0.05,  # accuracy should be at least 5 percent greater than baseline model accuracy
"""
thresholds = {
    "accuracy_score": MetricThreshold(
        threshold=0.83,
        greater_is_better=True,
    ),
}

# Provide an Experiment description that will appear in the UI
experiment_description = (
    "This is the Heart disease prediction project. " "This experiment contains the produce models for Heart Disease."
)

# Provide searchable tags that define characteristics of the Runs that
# will be in this Experiment
experiment_tags = {
    "project_name": "heart-disease-forecasting",
    "store_dept": "health",
    "team": "stores-ml",
    "project_quarter": "Q4-2024",
    "mlflow.note.content": "This is the Heart disease prediction project.",
}


@task
def train_and_log_model(retries=20, retry_delay_seconds=10):
    """Train model and log metrics and artifacts in MLflow."""
    experiment_name = "Health Models Experiment 0"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(name=experiment_name, tags=experiment_tags)
    mlflow.set_experiment(experiment_name)

    # Load and prepare data
    df = pd.read_csv("data/heart.csv")

    X = df.drop(["target"], axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Initialize preprocessor and model pipeline

    # Initialisation du preprocesseur
    preprocessor = DataPreprocessor()

    # Créer un pipeline avec le préprocesseur et le modèle

    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "bootstrap": True,
        "oob_score": False,
        "random_state": 42,
    }
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(**params)),
        ]
    )

    # Entraîner le modèle
    pipeline.fit(X_train, y_train)

    # Prédire et calculer les métriques
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Plot ROC curve and confusion matrix
    fpr, tpr, _ = roc_curve(y_test, pipeline.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)

    # # Créer un répertoire temporaire pour stocker les artefacts
    artifacts_path = "artifacts"
    os.makedirs(artifacts_path, exist_ok=True)

    # Enregistrer la courbe ROC comme artefact
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    roc_path = os.path.join(artifacts_path, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()

    # Save confusion matrix as artifact
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
    plt.title("Matrice de confusion")
    cm_path = os.path.join(artifacts_path, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # Save metrics to CSV
    metrics_df = pd.DataFrame({"accuracy": [accuracy], "f1_score": [f1]})
    metrics_csv_path = os.path.join(artifacts_path, "metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)

    # Define the model signature
    signature = infer_signature(X_train, y_train)

    # Enregistrement des artefacts dans MLflow
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Enregistrer les artefacts dans MLflow
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(metrics_csv_path)

        # Evaluation du modèle avec MLflow avec  le baseline défini
        eval_data = X_test.copy()
        eval_data["target"] = y_test

        evaluate_and_compare(run, artifacts_path, candidate_model=pipeline, eval_data=eval_data, signature=signature)

    logger = get_run_logger()
    logger.info(f"Model trained successfully: Accuracy={accuracy}, F1 Score={f1}")

    if accuracy < thresholds["accuracy_score"].threshold:
        raise ValueError(
            f"Accuracy {accuracy:.2f} below threshold {thresholds['accuracy_score'].threshold}, retry triggered."
        )


@task
def evaluate_and_compare(run, artifact_path, candidate_model, eval_data, signature):
    # Enregistrer le modèle dans le Model Registry
    model_name = "RandomForestHeartDiseaseModel1"
    model_uri = mlflow.sklearn.log_model(
        sk_model=candidate_model, artifact_path=artifact_path, signature=signature
    ).model_uri

    mlflow.register_model(model_uri, model_name)

    if baseline_model_uri:
        evaluation_result: EvaluationResult = mlflow.evaluate(
            model=run.info.artifact_uri + "/artifacts",
            data=eval_data,
            targets="target",
            model_type="classifier",
            validation_thresholds=thresholds,
            baseline_model=baseline_model_uri,  # Set the baseline model for comparison
        )
        print("Evaluation Result:", evaluation_result.metrics)


@flow(name="data-quality-training-pipeline")
def main_flow():
    train_and_log_model()


# Exécuter le flux
if __name__ == "__main__":
    main_flow()
