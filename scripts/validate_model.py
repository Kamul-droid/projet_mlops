from typing import Any, Dict

import mlflow
from mlflow.models import convert_input_example_to_serving_input, validate_serving_input

# Constantes
MLFLOW_TRACKING_URI: str = "http://localhost:5000"
MODEL_URI: str = "runs:/e095852ec5f64e89894a61edb466ce8e/artifacts"

# Configuration de l'URI de suivi MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def generate_serving_payload(input_example: Dict[str, Any]) -> Dict[str, Any]:
    """Génère une charge utile (payload) de prédiction pour tester l'entrée du modèle avant le déploiement.

    Args:
        input_example (Dict[str, Any]): Un exemple de données d'entrée pour le modèle.

    Returns:
        Dict[str, Any]: Données transformées prêtes pour l'inférence.
    """
    return convert_input_example_to_serving_input(input_example)


def validate_model_serving_payload(model_uri: str, payload: Dict[str, Any]) -> None:
    """Valide la charge utile de prédiction avec le modèle pour vérifier que l'entrée est correcte.

    Args:
        model_uri (str): URI du modèle dans MLflow.
        payload (Dict[str, Any]): Charge utile pour le test d'inférence.

    Returns:
        None: Affiche si la validation a réussi ou si des erreurs sont présentes.
    """
    validate_serving_input(model_uri, payload)
    print("Validation réussie pour la charge utile du modèle.")


if __name__ == "__main__":
    # Exemple de données d'entrée pour le modèle
    INPUT_EXAMPLE: Dict[str, Any] = {
        "age": [55],
        "sex": [1],
        "cp": [2],
        "trestbps": [130],
        "chol": [250],
        "fbs": [0],
        "restecg": [1],
        "thalach": [160],
        "exang": [0],
        "oldpeak": [2.3],
        "slope": [2],
        "ca": [0],
        "thal": [1],
    }

    # Générer et valider la charge utile de prédiction pour le modèle
    serving_payload: Dict[str, Any] = generate_serving_payload(INPUT_EXAMPLE)
    validate_model_serving_payload(MODEL_URI, serving_payload)
