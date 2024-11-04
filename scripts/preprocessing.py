import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from typing import Tuple

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