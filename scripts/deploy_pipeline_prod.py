import os
from typing import Any

import cloudpickle
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from mlflow.models import infer_signature
from prefect import flow, get_run_logger, task
from preprocessing import DataPreprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, roc_curve
from sklearn.model_selection import train_test_split

# Configurer l'URI de suivi de MLflow
MLFLOW_TRACKING_URI: str = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)

# Description de l'expérience pour l'interface MLflow
EXPERIMENT_DESCRIPTION: str = (
    "This is the Heart disease prediction project. " "This experiment contains the produce models for Heart Disease."
)
EXPERIMENT_TAGS = {
    "project_name": "heart-disease-forecasting",
    "store_dept": "health",
    "team": "stores-ml",
    "project_quarter": "Q4-2024",
    "mlflow.note.content": EXPERIMENT_DESCRIPTION,
}


@task
def train_and_log_model():
    """Entraîne le modèle et enregistre les métriques et artefacts dans MLflow."""

    experiment_name = "Model created with external preprocessing"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name, tags=EXPERIMENT_TAGS)
    mlflow.set_experiment(experiment_name)

    # Chargement des données nettoyées pour l'entraînement et le test
    X_train = pd.read_csv("data/X_train_clean.csv")
    X_test = pd.read_csv("data/X_test_clean.csv")
    y_train = pd.read_csv("data/y_train.csv").squeeze()
    y_test = pd.read_csv("data/y_test.csv").squeeze()

    # Préparation et entraînement du préprocesseur
    df = pd.read_csv("data/heart.csv")
    X = df.drop(["target"], axis=1)
    y = df["target"]
    X_train_p, _, _, _ = train_test_split(X, y, test_size=0.2, stratify=y)
    preprocessor = DataPreprocessor()
    _ = preprocessor.fit_transform(X_train_p)

    # Initialisation et entraînement du modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prédictions et calcul des métriques
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)

    # Création du répertoire pour les artefacts
    artifacts_path = "artifacts"
    os.makedirs(artifacts_path, exist_ok=True)

    # Sauvegarde de la courbe ROC
    roc_path = save_roc_curve(fpr, tpr, roc_auc, artifacts_path)

    # Sauvegarde de la matrice de confusion
    cm_path = save_confusion_matrix(cm, artifacts_path)

    # Sauvegarde des métriques dans un fichier CSV
    metrics_csv_path = save_metrics_csv(accuracy, f1, artifacts_path)

    # Enregistrement des artefacts dans MLflow
    with mlflow.start_run() as run:
        log_metrics_and_params(model, accuracy, f1, roc_path, cm_path, metrics_csv_path)
        save_preprocessor(preprocessor)
        eval_data = prepare_evaluation_data(X_test, y_test)
        signature = infer_signature(X_train, y_train)
        evaluate_and_compare(run, artifacts_path, model, eval_data, signature)

    logger = get_run_logger()
    logger.info(f"Modèle entraîné avec succès : Accuracy={accuracy}, F1 Score={f1}")


def save_roc_curve(fpr, tpr, roc_auc, artifacts_path: str) -> str:
    """Génère et sauvegarde la courbe ROC en tant qu'artefact."""
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    roc_path = f"{artifacts_path}/roc_curve.png"
    plt.savefig(roc_path)
    plt.close()
    return roc_path


def save_confusion_matrix(cm, artifacts_path: str) -> str:
    """Génère et sauvegarde la matrice de confusion en tant qu'artefact."""
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
    cm_path = f"{artifacts_path}/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    return cm_path


def save_metrics_csv(accuracy: float, f1: float, artifacts_path: str) -> str:
    """Sauvegarde les métriques sous forme de fichier CSV."""
    metrics_df = pd.DataFrame({"accuracy": [accuracy], "f1_score": [f1]})
    metrics_csv_path = f"{artifacts_path}/metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    return metrics_csv_path


def log_metrics_and_params(model, accuracy: float, f1: float, roc_path: str, cm_path: str, metrics_csv_path: str):
    """Enregistre les paramètres, métriques et artefacts dans MLflow."""
    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_artifact(roc_path)
    mlflow.log_artifact(cm_path)
    mlflow.log_artifact(metrics_csv_path)


def save_preprocessor(preprocessor: Any):
    """Sauvegarde le préprocesseur en tant qu'artefact MLflow."""
    with open("preprocessor.pkl", "wb") as f:
        cloudpickle.dump(preprocessor, f)
    mlflow.log_artifact("preprocessor.pkl")


def prepare_evaluation_data(X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Prépare les données pour l'évaluation du modèle."""
    eval_data = X_test.copy()
    eval_data["target"] = y_test
    return eval_data


@task
def evaluate_and_compare(run, artifact_path: str, candidate_model: Any, eval_data: pd.DataFrame, signature: Any):
    """Évalue le modèle candidat et compare avec les valeurs de référence."""
    model_name = "RandomForestHeartDiseasesPredictionsModel"
    model_uri = mlflow.sklearn.log_model(
        sk_model=candidate_model, artifact_path=artifact_path, signature=signature
    ).model_uri
    mlflow.register_model(model_uri, model_name)
    # Comparaison avec le modèle de référence pourrait être ajouté ici si applicable


@flow(name="model-training-pipeline")
def main_flow():
    train_and_log_model()


if __name__ == "__main__":
    main_flow()
