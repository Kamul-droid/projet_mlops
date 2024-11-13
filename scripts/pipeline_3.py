import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from mlflow.models import EvaluationResult, MetricThreshold, infer_signature
from prefect import flow, get_run_logger, task
from prefect.server.events.triggers import evaluate
from preprocessing import DataPreprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000") 


#  Load the model from the previous experiment run
logged_model = 'runs:/b4ae7db152774ffd90448a0ec3b1fdda/model'

baseline_model_uri = logged_model

# Define criteria for model to be validated against
thresholds = {
    "accuracy_score": MetricThreshold(
        threshold=0.8,  # accuracy should be >=0.8
        # min_absolute_change=0.05,  # accuracy should be at least 0.05 greater than baseline model accuracy
        # min_relative_change=0.05,  # accuracy should be at least 5 percent greater than baseline model accuracy
        greater_is_better=True,
    ),
}

@task
def train_and_log_model():
    """Entraîne le modèle, enregistre les métriques et les artefacts dans MLflow."""
    experiment_name = "experience_3"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # Charger les données nettoyées
    df = pd.read_csv("data/heart.csv")
    
    X = df.drop(["target"], axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    
    
    # Initialisation du preprocesseur
    preprocessor = DataPreprocessor()
   
   
    # Créer un pipeline avec le préprocesseur et le modèle
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(n_estimators=50, random_state=42))
    ])
    
    
    # Entraîner le modèle
    pipeline.fit(X_train, y_train)
    
    # Prédire et calculer les métriques
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Calcul des courbes ROC et de la matrice de confusion
    fpr, tpr, _ = roc_curve(y_test, pipeline.predict_proba(X_test)[:, 1])
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

    # Définir la signature du modèle
    signature = infer_signature(X_train, y_train)
    
    
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
           
        # Evaluation du modèle avec MLflow avec  le baseline défini
        eval_data= X_test.copy()
        eval_data["target"] = y_test
       
        evaluate_and_compare(candidate_model=pipeline, eval_data=eval_data,signature=signature)
        
        
    logger = get_run_logger()
    logger.info(f"Modèle entraîné avec succès : Accuracy={accuracy}, F1 Score={f1}")


@task
# Evaluate new model and compare with the baseline
def evaluate_and_compare(candidate_model, eval_data, signature):
    # Enregistrer le modèle dans le Model Registry
    candidate_model_uri = mlflow.sklearn.log_model(
        sk_model=candidate_model,
        artifact_path="model",
        registered_model_name="RandomForest",
        signature=signature
    ).model_uri


    # Compare the candidate model with the baseline model
    evaluation_result: EvaluationResult = mlflow.evaluate(
        model=candidate_model_uri,
        data=eval_data,
        targets="target",
        model_type="classifier",
        validation_thresholds=thresholds,
        baseline_model=baseline_model_uri  # Set the baseline model for comparison
    )

    # Output results of comparison
    print("Evaluation Result:", evaluation_result.metrics)


@flow(name="data-quality-training-pipeline")
def main_flow():
    train_and_log_model()

# Exécuter le flux
if __name__ == "__main__":
    main_flow()
