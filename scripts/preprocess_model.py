import mlflow.pyfunc as mlflow_func
import pandas as pd
import os
import joblib


# Modèle MLflow personnalisé pour appliquer le prétraitement et effectuer des prédictions
class PreprocessingModel(mlflow_func.PythonModel):
    def load_context(self, context):
        """
        Charge le modèle et le préprocesseur à partir des artefacts MLflow.
        """
        self.model = joblib.load(context.artifacts["model_path"])
        self.preprocessor = joblib.load(context.artifacts["preprocessor_path"])

    def predict(self, context, model_input: pd.DataFrame) -> pd.Series:
        """
        Applique le prétraitement aux données d'entrée et effectue la prédiction.

        Args:
            context : Contexte d'exécution MLflow.
            model_input (pd.DataFrame): Données d'entrée brutes sous forme de DataFrame.

        Returns:
            pd.Series: Résultats de la prédiction.
        """
        # Transformation des données d'entrée
        processed_input = self.preprocessor.transform(model_input)
        # Extraction des noms de colonnes du préprocesseur
        transformed_columns = self.preprocessor.get_feature_names_out()
        processed_input_df = pd.DataFrame(processed_input, columns=transformed_columns)

        # Faire la prédiction
        return self.model.predict(processed_input_df)
