import pandas as pd
import yaml


def load_config() -> dict:
    """
    Charge le fichier de configuration pour obtenir les chemins de fichiers et autres paramètres.

    Returns:
        dict: Dictionnaire contenant la configuration chargée depuis le fichier YAML.
    """
    with open("../config/config.yaml", "r") as file:
        return yaml.safe_load(file)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Charge les données depuis un fichier CSV spécifié.

    Args:
        filepath (str): Chemin vers le fichier CSV.

    Returns:
        pd.DataFrame: DataFrame contenant les données chargées.
    """
    return pd.read_csv(filepath)


def save_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Sauvegarde le DataFrame dans un fichier CSV.

    Args:
        df (pd.DataFrame): Données à sauvegarder.
        filepath (str): Chemin vers le fichier de destination CSV.
    """
    df.to_csv(filepath, index=False)


def main() -> None:
    """
    Fonction principale pour l'ingestion des données, incluant chargement, sauvegarde,
    et configuration des chemins de fichier.
    """
    config = load_config()
    path =config["data_path"]
    print(path)
    data = load_data(f"../{path}")
    # data = load_data(config["data_path"])
    process_path=config["processed_data_path"]
    save_data(data, f"../{process_path}")
    print("Ingestion des données terminée et enregistrée pour les contrôles de qualité.")


if __name__ == "__main__":
    main()
