import great_expectations as ge
import pandas as pd
import yaml

def load_config():
    """Charge le fichier de configuration pour obtenir les chemins de données et les types attendus."""
    with open("config/config.yaml", "r") as file:
        return yaml.safe_load(file)

def check_null_values(df):
    """Vérifie le nombre de valeurs nulles par colonne."""
    return df.isnull().sum()

def check_duplicates(df):
    """Vérifie le nombre de doublons dans le DataFrame."""
    return df.duplicated().sum()

def validate_with_great_expectations(df, config):
    """Valide les types et valeurs de colonnes dans le DataFrame avec Great Expectations."""
    # Convertir le DataFrame en un objet Dataset pour Great Expectations
    ge_df = ge.dataset.PandasDataset(df)
    expected_types = config["expected_data_types"]
    
    results = {}

    # Vérifier les types de données pour chaque colonne
    for column, dtype in expected_types.items():
        expectation_result = ge_df.expect_column_values_to_be_of_type(column, dtype)
        results[f"{column}_type_check"] = expectation_result.success

    # Vérifier les valeurs de "age" et "sex" avec des bornes spécifiques
    age_check = ge_df.expect_column_values_to_be_between("age", min_value=1, max_value=120)
    sex_check = ge_df.expect_column_values_to_be_in_set("sex", [0, 1])
    
    results["age_value_check"] = age_check.success
    results["sex_value_check"] = sex_check.success

    return results

def main():
    """Fonction principale pour exécuter toutes les vérifications de qualité."""
    config = load_config()
    df = pd.read_csv(config["processed_data_path"])

    # Vérification des valeurs nulles
    null_values = check_null_values(df)
    print("Valeurs nulles:\n", null_values)

    # Vérification des doublons
    duplicates = check_duplicates(df)
    print("Doublons:\n", duplicates)

    # Validation des données avec Great Expectations
    validation_results = validate_with_great_expectations(df, config)
    print("Résultats des contrôles de qualité avec Great Expectations:\n", validation_results)

if __name__ == "__main__":
    main()