import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from prefect import flow, get_run_logger, task
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             roc_curve)


@task
def train_and_log_model():
    """Entraîne le modèle, enregistre les métriques et les artefacts dans MLflow."""
    mlflow.set_tracking_uri("mlflow_run")  # Remplacez par le chemin réel
    experiment_name = "experience_1"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # Charger les données nettoyées
    X_train = pd.read_csv("data/X_train_clean.csv")
    X_test = pd.read_csv("data/X_test_clean.csv")
    y_train = pd.read_csv("data/y_train.csv").squeeze()
    y_test = pd.read_csv("data/y_test.csv").squeeze()
    
    # Initialiser et entraîner le modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Prédire et calculer les métriques
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Calcul des courbes ROC et de la matrice de confusion
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    cm = confusion_matrix(y_test, y_pred)
    
    # Créer un répertoire temporaire pour stocker les artefacts
    artifacts_path = "artifacts"
    os.makedirs(artifacts_path, exist_ok=True)
    
    # Enregistrer la courbe ROC comme artefact
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

    # Enregistrer la matrice de confusion comme artefact
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
    plt.title("Matrice de confusion")
    cm_path = os.path.join(artifacts_path, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # Enregistrer les métriques dans un fichier CSV
    metrics_df = pd.DataFrame({"accuracy": [accuracy], "f1_score": [f1]})
    metrics_csv_path = os.path.join(artifacts_path, "metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)

    # Enregistrement des artefacts dans MLflow
    with mlflow.start_run() as run:
        run_id = run.info.run_id  # Capture du run_id
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        # Enregistrer les artefacts dans MLflow
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(metrics_csv_path)
        
        # Enregistrer le modèle dans le Model Registry
        model_name = "RandomForest_Heart_Model"
        mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)

    logger = get_run_logger()
    logger.info(f"Modèle entraîné avec succès : Accuracy={accuracy}, F1 Score={f1}")

@flow(name="data-quality-training-pipeline")
def main_flow():
    train_and_log_model()

# Exécuter le flux
if __name__ == "__main__":
    main_flow()
