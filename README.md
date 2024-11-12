# Heart Attack Data Pipeline

## Description
Ce projet implémente un pipeline pour l'ingestion, le contrôle de qualité et le prétraitement de données de crises cardiaques. Il utilise **Great Expectations** pour des contrôles de qualité avancés, assurant que les données sont conformes aux types attendus et aux bornes de valeurs pour des variables critiques telles que l'âge et le sexe.

## Prérequis
- Python 3.x
- Poetry 
- Installez les dépendances via 
  -`poetry shell`
  -`poetry install` (Depuis la racine du projet; executer cette commande. Elle crée un environnement de travail virtuel avec toutes les dépendances)

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
1. Ouvrir un shell poetry dans le dossier scripts et lancer le serveur FastAPI avec la commande :
   ```shell poetry
   uvicorn server:app --reload
   ```


### Notebook pour le preprocessing
```
/project-directory
│
├── /data                    # Fichiers de données : X_train_clean.csv, X_test_clean.csv, y_train.csv, y_test.csv
├── /scripts                 # Scripts des pipelines et autres tâches
├── main.ipynb               # Notebook pour le preprocessing

```


 ```poetry shell
poetry run jupyter lab --port=6200
```



## Orchestration des Pipelines et Suivi des Expériences avec Prefect et MLflow

Dans cette partie du projet, nous mettons en place des pipelines de machine learning utilisant **Prefect** pour l'orchestration des tâches et **MLflow** pour le suivi des expériences et la gestion des artefacts. Cette section comprend l'entraînement des modèles de régression logistique et de RandomForest, ainsi que l'intégration des résultats dans un système de gestion centralisé pour le suivi des performances.
### Structure dossier du projet
Le projet est organisé en plusieurs dossiers et fichiers pour une gestion optimale des tâches. 
Voici la structure du projet :

```
/project-directory
│
├── /data                    # Fichiers de données : X_train_clean.csv, X_test_clean.csv, y_train.csv, y_test.csv
├── /scripts                 # Scripts des pipelines et autres tâches
│   ├── pipeline_2.py        # Pipeline avec régression logistique
│   ├── pipeline_3.py        # Pipeline avec RandomForest
│   ├── data_loader.py       # Chargement des données
│   ├── preprocessing.py     # Prétraitement des données
│   ├── quality_checks.py    # Contrôle de la qualité des données
│   └── server.py            # Serveur pour exécution des workflows
├── /artifacts               # Artefacts générés : graphes, métriques, modèles
├── /great_expectations      # Vérification de la qualité des données (expectations)
├── /mlflow_run              # Répertoire pour stocker les métadonnées des expériences MLflow
│   ├── 0                    # Contient les logs et les résultats des exécutions MLflow
│   ├── models               # Contient les logs et les résultats des models MLflow
└── config.yaml              # Fichier de configuration pour l'orchestration
```

### Insrallation des dépendances 



### Démarrage de l'interface Prefect (Prefect UI)
L’interface graphique de Prefect permet de visualiser vos workflows, suivre leur exécution, et gérer les tâches en cours. Pour y accéder, il faut démarrer le serveur Prefect UI localement.

Pour démarrer le serveur Prefect UI, exécuter cette commande dans le terminal (CMD) :2.17.2

  ```poetry shell
prefect server start
   ```

Pour demarrer les services de prefect :
- Un serveur GraphQL pour interagir avec les flows
- Un serveur de base de données pour stocker les logs et les états d'exécution
- Une interface web (Prefect UI) pour visualiser et gérer les workflows
L’interface web sera disponible à l'adresse suivante : http://localhost:4200


### Démarrage de MLflow UI
scripts/mlfow_run est le répertoire dans lequel les expériences et les artefacts et les mmodéles seront sauvegardés
Ouvrir un shell poetry avec la  CMD `poetry shell` à la racine du projet :
  ```poetry shell
 mlflow ui --backend-store-uri scripts/mlflow_run
   ```

 Cette commmande va démarrer le serveur MLflow à l'adresse : http://localhost:5000. Visualiser les logs, les métriques et les artefacts générés par les exécutions des pipelines.
 
  ### Exécution des pipelines Prefect
Une fois MLflow UI démarré, maintenant il faut exécuter les pipelines pour entraîner des modèles et suivre leur exécution dans MLflow.
 **- pipeline de régression logistique**
Pour entraîner le modèle de **Logistic_regression**, exécuter la commande suivante à partir de la racine du projet:

  ```poetry shell
python scripts/pipeline_2.py
   ```

**- pipeline de RandomForest**
Pour entraîner le  modèle de RandomForest, exécute la commande suivante à partir de la racine du projet :
  ```poetry shell
python scripts/pipeline_3.py
   ```

### Visualisation les résultats dans MLflow UI
Après l'exécution des pipelines, la visualisation des résultats peut être faite dans MLflow UI. Accéder à l'interface MLflow à l'adresse suivante : http://localhost:5000.
Pour consulter :
Accéder à Expérience 1
Metrics : **précision** et le **F1 Score** des modèles.
Artifacts : Les artefacts ici sont: les courbes ROC, les matrices de confusion et les modèles enregistrés.
