from typing import Tuple
from typing import Optional
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
import mlflow.pyfunc
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin 
import mlflow.pyfunc


# Definition de la catégorie des caractéristiques
num_variables = ["age", "trestbps", "chol", "thalach", "ca", "oldpeak"]
cat_variables = ["cp", "restecg", "thal"]
ord_variables = ["slope"]
bin_variables = ["sex", "fbs", "exang"]

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prépare les données en appliquant diverses transformations et en les séparant en ensembles d'entraînement et de test.

    Args:
        df (pd.DataFrame): Données brutes contenant les variables à traiter.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Données transformées pour l'entraînement et les tests, ainsi que les étiquettes correspondantes.
    """
    X = df.drop(["target"], axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=df["target"])
    
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer()),
        ("normalization", MinMaxScaler())
    ])
    
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first"))
    ])
    
    ord_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder())
    ])
    
    bin_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", num_pipeline, num_variables),
        ("categorical", cat_pipeline, cat_variables),
        ("ordinal", ord_pipeline, ord_variables),
        ("binary", bin_pipeline, bin_variables),
    ])
    
    X_train_clean = preprocessor.fit_transform(X_train)
    X_test_clean = preprocessor.transform(X_test)
    
    return pd.DataFrame(X_train_clean), pd.DataFrame(X_test_clean), y_train, y_test


# Define the preprocessing function as a custom transformer
class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Initialise le DataPreprocessor avec des pipelines spécifiques pour 
        le prétraitement des données numériques, catégorielles, ordinales et binaires.
        Chaque pipeline est configuré avec des transformations appropriées 
        telles que l'imputation et la normalisation/encodage.
        """
        
        # Définir le pipeline de prétraitement pour les variables numériques
        self.num_pipeline = Pipeline([
            ("imputer", SimpleImputer()),  # Remplit les valeurs manquantes avec la moyenne (stratégie par défaut)
            ("normalization", MinMaxScaler())  # Normalise les caractéristiques entre [0, 1]
        ])
        
        # Définir le pipeline de prétraitement pour les variables catégorielles
        self.cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),  # Remplit les valeurs manquantes avec la valeur la plus fréquente
            ("encoder", OneHotEncoder(drop="first"))  # Applique un encodage one-hot et retire la première catégorie pour éviter la colinéarité
        ])
        
        # Définir le pipeline de prétraitement pour les variables ordinales
        self.ord_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),  # Remplit les valeurs manquantes avec la valeur la plus fréquente
            ("encoder", OrdinalEncoder())  # Encode les catégories en valeurs ordinales
        ])
        
        # Définir le pipeline de prétraitement pour les variables binaires
        self.bin_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent"))  # Remplit les valeurs manquantes avec la valeur la plus fréquente
        ])
        
        # Combiner tous les pipelines dans un seul ColumnTransformer
        self.preprocessor = ColumnTransformer(transformers=[
            ("numeric", self.num_pipeline, num_variables),
            ("categorical", self.cat_pipeline, cat_variables),
            ("ordinal", self.ord_pipeline, ord_variables),
            ("binary", self.bin_pipeline, bin_variables),
        ])

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DataPreprocessor":
        """
        Entraîne le ColumnTransformer sur le DataFrame d'entrée.
        
        Args:
            X (pd.DataFrame): Données d'entrée à utiliser pour l'entraînement.
            y (pd.Series, optional): Valeurs cibles (non utilisées, pour compatibilité avec scikit-learn).
        
        Returns:
            DataPreprocessor: Instance entraînée de la classe.
        """
        # Entraîner le préprocesseur sur les caractéristiques d'entrée
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforme le DataFrame d'entrée en utilisant le ColumnTransformer entraîné.
        
        Args:
            X (pd.DataFrame): Données d'entrée à transformer.
        
        Returns:
            pd.DataFrame: Données transformées sous forme de DataFrame.
        """
        # Appliquer les transformations et retourner le résultat sous forme de DataFrame
        return pd.DataFrame(self.preprocessor.transform(X))
    
    def get_feature_names_out(self):
        """Récupère les noms des colonnes après transformation."""
        return self.preprocessor.get_feature_names_out()