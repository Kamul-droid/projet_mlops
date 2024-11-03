# Heart Attack Data Pipeline

## Description
Ce projet implémente un pipeline pour l'ingestion, le contrôle de qualité et le prétraitement de données de crises cardiaques. Il utilise **Great Expectations** pour des contrôles de qualité avancés, assurant que les données sont conformes aux types attendus et aux bornes de valeurs pour des variables critiques telles que l'âge et le sexe.

## Prérequis
- Python 3.x
- Installez les dépendances via `pip install -r requirements.txt`

## Configuration
Le fichier `config/config.yaml` permet de configurer :
- **data_path** : Chemin du fichier de données brut (par défaut, `data/heart.csv`).
- **processed_data_path** : Chemin pour sauvegarder les données transformées (par défaut, `data/processed_heart.csv`).
- **expected_data_types** : Dictionnaire définissant les types de données attendus pour chaque colonne.

## Modules Principaux
1. **data_loader.py** : Module d'ingestion de données.
2. **quality_checks.py** : Module de contrôle de qualité utilisant **Great Expectations**.
3. **preprocessing.py** : Module de prétraitement pour normaliser et encoder les données.
4. **server.py** : API pour lancer les processus via des endpoints HTTP avec **FastAPI**.

## Utilisation
1. Lancer le serveur FastAPI avec la commande :
   ```bash
   uvicorn scripts.server:app --reload
   ```
