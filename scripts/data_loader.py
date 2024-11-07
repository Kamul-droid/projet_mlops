import pandas as pd
import yaml

def load_config():
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)

def load_data(filepath):
    return pd.read_csv(filepath)

def save_data(df, filepath):
    df.to_csv(filepath, index=False)

def main():
    config = load_config()
    data = load_data(config["data_path"])
    save_data(data, config["processed_data_path"])
    print("Ingestion des données terminée et enregistrée pour les contrôles de qualité.")

if __name__ == "__main__":
    main()