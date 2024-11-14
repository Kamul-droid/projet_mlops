import json
from typing import Any, Dict

import cloudpickle
import mlflow
import pandas as pd
import requests

# Constantes
MLFLOW_TRACKING_URI: str = "http://localhost:5000"
MODEL_URI: str = "runs:/a9f6bf7101724ab591340cb90b5aff2e/artifacts"
PREPROCESSOR_URI: str = "runs:/a9f6bf7101724ab591340cb90b5aff2e/preprocessor.pkl"
PREDICTION_SERVER_URL: str = "http://localhost:5001/invocations"

# Définir l'URI de suivi MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_model(model_uri: str) -> mlflow.pyfunc.PyFuncModel:
    """Charge le modèle entraîné depuis MLflow.
    
    Args:
        model_uri (str): URI du modèle dans MLflow.

    Returns:
        mlflow.pyfunc.PyFuncModel: Modèle MLflow chargé.
    """
    return mlflow.pyfunc.load_model(model_uri=model_uri)

def load_preprocessor(preprocessor_uri: str) -> Any:
    """Charge l'objet de prétraitement depuis les artefacts MLflow.
    
    Args:
        preprocessor_uri (str): URI de l'artefact du préprocesseur.

    Returns:
        Any: Préprocesseur chargé, typiquement un transformer scikit-learn.
    """
    preprocessor_path: str = mlflow.artifacts.download_artifacts(preprocessor_uri)
    with open(preprocessor_path, "rb") as f:
        return cloudpickle.load(f)

def preprocess_data(preprocessor: Any, data: pd.DataFrame) -> pd.DataFrame:
    """Applique le prétraitement aux données d'entrée brutes.
    
    Args:
        preprocessor (Any): Préprocesseur chargé.
        data (pd.DataFrame): Données brutes à transformer.

    Returns:
        pd.DataFrame: Données transformées prêtes pour l'inférence.
    """
    return preprocessor.transform(data)

def make_prediction(model: mlflow.pyfunc.PyFuncModel, data: pd.DataFrame) -> pd.Series:
    """Effectue une prédiction avec le modèle chargé.
    
    Args:
        model (mlflow.pyfunc.PyFuncModel): Modèle chargé pour l'inférence.
        data (pd.DataFrame): Données prétraitées pour la prédiction.

    Returns:
        pd.Series: Résultats de la prédiction.
    """
    return model.predict(data)

def send_request_to_model_server(preprocessed_data: pd.DataFrame, url: str) -> Dict:
    """Envoie une requête POST contenant les données prétraitées au serveur de modèle.
    
    Args:
        preprocessed_data (pd.DataFrame): Données prétraitées au format DataFrame.
        url (str): URL du serveur de modèle.

    Returns:
        Dict: Réponse JSON du serveur contenant les prédictions.
    """
    data_json: str = json.dumps({"dataframe_split": preprocessed_data.to_dict(orient="split")})
    response = requests.post(url, headers={"Content-Type": "application/json"}, data=data_json)
    return response.json()

# Exécution principale
if __name__ == "__main__":
    # Charger le modèle et le préprocesseur
    loaded_model = load_model(MODEL_URI)
    preprocessor = load_preprocessor(PREPROCESSOR_URI)
    
    # Exemple de données d'entrée
    example_data: pd.DataFrame = pd.DataFrame({
        "age": [63],
        "sex": [1],
        "cp": [1],
        "trestbps": [145],
        "chol": [233],
        "fbs": [0],
        "restecg": [0],
        "thalach": [150],
        "exang": [0],
        "oldpeak": [2.3],
        "slope": [2],
        "ca": [0],
        "thal": [1]
    })

    # Prétraiter les données
    preprocessed_data: pd.DataFrame = preprocess_data(preprocessor, example_data)
    preprocessed_data.columns = preprocessed_data.columns.astype(str)

    # Prédiction locale avec le modèle
    local_prediction = make_prediction(loaded_model, preprocessed_data)
    print("Prédiction locale :", local_prediction)

    # Prédiction serveur avec modèle déployé
    server_prediction = send_request_to_model_server(preprocessed_data, PREDICTION_SERVER_URL)
    print("Prédiction serveur :", server_prediction)
