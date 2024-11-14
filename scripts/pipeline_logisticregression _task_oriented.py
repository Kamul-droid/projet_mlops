import data_loader
import mlflow
import mlflow.sklearn
import pandas as pd
import preprocessing
import quality_checks
from prefect import flow, get_run_logger, task
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)


@task
def ingest_data():
    """Ingestion de données brutes."""
    data_loader.main()
    logger = get_run_logger()
    logger.info("Données ingérées avec succès.")


@task
def run_quality_checks():
    """Exécution des contrôles de qualité avec Great Expectations."""
    quality_checks.main()
    logger = get_run_logger()
    logger.info("Contrôles de qualité réussis.")


@task
def preprocess_data():
    """Prétraitement des données après contrôle de qualité."""
    df = pd.read_csv(r"data/processed_heart.csv")
    X_train, X_test, y_train, y_test = preprocessing.preprocess_data(df)

    X_train.to_csv("data\X_train_clean.csv", index=False)
    X_test.to_csv("data\X_test_clean.csv", index=False)
    pd.DataFrame(y_train).to_csv("data\y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv("data\y_test.csv", index=False)

    logger = get_run_logger()
    logger.info("Prétraitement des données terminé.")


@task
def train_and_log_model():
    """Entraîne le modèle et enregistre les métriques dans MLflow."""
    mlflow.set_tracking_uri("mlflow_run")  # Remplacez par le chemin réel
    experiment_name = "experience_LR"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # Charger les données nettoyées
    X_train = pd.read_csv(r"data/X_train_clean.csv")
    X_test = pd.read_csv(r"data/X_test_clean.csv")
    y_train = pd.read_csv(r"data/y_train.csv").squeeze()
    y_test = pd.read_csv(r"data/y_test.csv").squeeze()

    # Initialiser le modèle Logistic Regression avec les paramètres spécifiés
    model = LogisticRegression(C=0.48, max_iter=10000, penalty="l1", solver="liblinear")

    # Entraîner le modèle
    model.fit(X_train, y_train)

    # Prédire et calculer les métriques
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilités pour la classe positive

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)

    # Enregistrer les résultats dans MLflow
    with mlflow.start_run() as run:
        run_id = run.info.run_id  # Capture du run_id
        mlflow.log_param("C", 0.48)
        mlflow.log_param("max_iter", 10000)
        mlflow.log_param("penalty", "l1")
        mlflow.log_param("solver", "liblinear")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", auc_roc)
        mlflow.log_metric("log_loss", logloss)

        # Enregistrement du modèle dans le Model Registry
        model_name = "LogisticRegression_Heart_Model"
        mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
        mlflow.register_model(f"runs:/{run_id}/model", model_name)

    logger = get_run_logger()
    logger.info(
        f"Modèle entraîné avec succès : Accuracy={accuracy}, F1 Score={f1}, Precision={precision}, Recall={recall}, AUC-ROC={auc_roc}, Log Loss={logloss}"
    )


@flow(name="data-quality-training-with-sub-task-pipeline")
def main_flow():
    raw_data = ingest_data()
    quality_checked = run_quality_checks(wait_for=[raw_data])
    preprocessed = preprocess_data(wait_for=[quality_checked])
    train_and_log_model(wait_for=[preprocessed])


# Exécuter le flux
if __name__ == "__main__":
    main_flow()
