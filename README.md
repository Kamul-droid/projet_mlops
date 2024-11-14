# Heart Attack Data Pipeline

## Description
Ce projet implémente un pipeline pour l'ingestion, le contrôle de qualité et le prétraitement de données de crises cardiaques. Il utilise **Great Expectations** pour des contrôles de qualité avancés, assurant que les données sont conformes aux types attendus et aux bornes de valeurs pour des variables critiques telles que l'âge et le sexe.

## Prérequis
- Python 3.x
- Docker Desktop

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
py -m poetry run jupyter lab --port=6200
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




### Installation des dépendances et démarrage des services

Le projet utilise Docker pour simplifier l’installation et le déploiement. Suivez les instructions ci-dessous pour construire l'image Docker et démarrer les services nécessaires.

Depuis le répertoire racine du projet, exécutez la commande suivante pour construire l'image Docker :

  ```poetry shell
docker build -t heart-attack-data-pipeline .
   ```

   Une fois l'image construite, lancez un conteneur avec la commande suivante :

```poetry shell
docker run -it --rm -p 8000:8000 -p 6200:6200 -p 5000:5000 -p 4200:4200 -p 3000:3000 -v "$(pwd):/app" heart-attack-data-pipeline
   ```



Cette commande démarre les services suivants :

Prefect UI (accessible à l'adresse : http://localhost:4200)
MLflow UI (accessible à l'adresse : http://localhost:5000)


 
  ### Exécution des pipelines Prefect

Une fois MLflow UI démarré,  il faut exécuter les pipelines pour entraîner des modèles et suivre leur exécution dans MLflow. 

Utilisez la commande suivante pour obtenir le nom ou l’ID du conteneur :

```
docker ps
```


Une fois que vous avez le nom du conteneur, exécutez la commande suivante pour ouvrir un shell :

```
docker exec -it [nom_du_conteneur] /bin/sh
docker exec -it [nom_du_conteneur] /bin/sh
```



 **- pipeline de régression logistique**
Puis pour entraîner le modèle de **Logistic_regression**, exécuter la commande suivante :

  ```poetry shell
poetry run python scripts/pipeline_logistic regression.py

   ```




**- pipeline de RandomForest**
Puis pour entraîner le  modèle de RandomForest, exécute la commande suivante :
  ```poetry shell
poetry run python scripts/pipeline_random forest.py

   ```


