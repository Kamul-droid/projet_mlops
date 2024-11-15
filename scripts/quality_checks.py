from typing import Dict, Tuple

import great_expectations as ge
import pandas as pd
import yaml


def load_config() -> dict:
    """
    Charge le fichier de configuration pour obtenir les chemins de données et les types attendus.

    Returns:
        dict: Dictionnaire contenant la configuration des données et les types attendus.
    """
    with open("../config/config.yaml", "r") as file:
        return yaml.safe_load(file)


def check_null_values(df: pd.DataFrame) -> pd.Series:
    """
    Vérifie le nombre de valeurs nulles par colonne.

    Args:
        df (pd.DataFrame): DataFrame contenant les données à vérifier.

    Returns:
        pd.Series: Série indiquant le nombre de valeurs nulles par colonne.
    """
    return df.isnull().sum()


def check_duplicates(df: pd.DataFrame) -> int:
    """
    Vérifie le nombre de doublons dans le DataFrame.

    Args:
        df (pd.DataFrame): DataFrame contenant les données à vérifier.

    Returns:
        int: Nombre de doublons trouvés.
    """
    return df.duplicated().sum()


def validate_with_great_expectations(df: pd.DataFrame, config: dict) -> Dict[str, bool]:
    """
    Valide les types et valeurs de colonnes dans le DataFrame avec Great Expectations.

    Args:
        df (pd.DataFrame): DataFrame contenant les données à valider.
        config (dict): Configuration spécifiant les types attendus pour les colonnes.

    Returns:
        Dict[str, bool]: Résultats de validation, indiquant le succès pour chaque contrôle.
    """
    ge_df = ge.dataset.PandasDataset(df)
    expected_types = config["expected_data_types"]

    results = {}

    for column, dtype in expected_types.items():
        expectation_result = ge_df.expect_column_values_to_be_of_type(column, dtype)
        results[f"{column}_type_check"] = expectation_result.success

    age_check = ge_df.expect_column_values_to_be_between("age", min_value=1, max_value=120)
    sex_check = ge_df.expect_column_values_to_be_in_set("sex", [0, 1])
    chol_check = ge_df.expect_column_values_to_be_between("chol", min_value=100, max_value=500)
    fbs_check = ge_df.expect_column_values_to_be_in_set("fbs", [0, 1])
    trestbps_check = ge_df.expect_column_values_to_be_between("trestbps", min_value=80, max_value=200)

    results["age_value_check"] = age_check.success
    results["sex_value_check"] = sex_check.success
    results["chol_value_check"] = chol_check.success
    results["fbs_value_check"] = fbs_check.success
    results["trestbps_value_check"] = trestbps_check.success

    return results


def delete_outliers(df: pd.DataFrame, thresholds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """
    Supprime les enregistrements du DataFrame qui dépassent les seuils définis pour certaines colonnes.

    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        thresholds (Dict[str, Tuple[float, float]]): Dictionnaire où les clés sont les noms de colonnes
            et les valeurs sont des tuples représentant les bornes minimum et maximum pour chaque colonne.

    Returns:
        pd.DataFrame: DataFrame sans les enregistrements qui dépassent les seuils spécifiés.
    """
    for column, (min_val, max_val) in thresholds.items():
        df = df[(df[column] >= min_val) & (df[column] <= max_val)]
    return df


def main() -> None:
    """
    Fonction principale pour exécuter toutes les vérifications de qualité sur les données.
    """
    config = load_config()
    path =config["processed_data_path"]
    df = pd.read_csv(f"../{path}")

    null_values = check_null_values(df)
    print("Valeurs nulles:\n", null_values)

    duplicates = check_duplicates(df)
    print("Doublons:\n", duplicates)

    validation_results = validate_with_great_expectations(df, config)
    print("Résultats des contrôles de qualité avec Great Expectations:\n", validation_results)

    thresholds = {"age": (1, 120), "chol": (100, 500), "fbs": (0, 1), "trestbps": (80, 200)}
    df_cleaned = delete_outliers(df, thresholds)
    print(f"Nombre d'enregistrements après suppression des outliers: {len(df_cleaned)}")


if __name__ == "__main__":
    main()
