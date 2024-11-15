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
Jupiter Lab UI (accessible à l'adresse : http://localhost:6200) # Utiliser le token afficher dans le terminal pour se connecter
Server FastApi  (accessible à l'adresse : http://localhost:8000) # Utiliser pour le data quality check



 
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
python scripts/pipeline_logistic_regression_one_flow.py
python scripts/pipeline_logisticregression_task_oriented.py
   ```
# Installation en local sur Windows


## Prérequis
- Python 3.x
- Poetry 
- Installez les dépendances via 
  -`poetry shell`
  -`poetry install` (Depuis la racine du projet; executer cette commande. Elle crée un environnement de travail virtuel avec toutes les dépendances)


## Utilisation
1. Ouvrir un shell poetry dans le dossier scripts et lancer le serveur FastAPI avec la commande :
   ```shell poetry
   uvicorn server:app --reload
   ```


### Repertoire du projet
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
 mlflow server  --host 0.0.0.0 --port 5000
   ```

 Cette commmande va démarrer le serveur MLflow à l'adresse : http://localhost:5000. Visualiser les logs, les métriques et les artefacts générés par les exécutions des pipelines.
 
  ### Exécution des pipelines Prefect
Avant d'exécuter les pipelines Prefect, exécuter cette commande : 
```poetry shell
prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
```
Une fois MLflow UI démarré, maintenant il faut exécuter les pipelines pour entraîner des modèles et suivre leur exécution dans MLflow.



# Architecture du Pipeline de Données
https://github.com/Kamul-droid/projet_mlops.git

## Objectif
Ce projet vise à créer un pipeline de données pour la préparation de données de crises cardiaques. Il garantit la qualité des données en utilisant des contrôles automatiques basés sur **Great Expectations**.

## Étapes du Pipeline
1. **Ingestion de Données** : Charge les données brutes depuis `data/heart.csv` et les enregistre dans `data/processed_heart.csv`.
2. **Contrôle de Qualité** : Exécute des contrôles de qualité avec **Great Expectations** pour valider les types de données, les doublons, les valeurs nulles, ainsi que la conformité des valeurs pour `age` et `sex`.
3. **Prétraitement des Données** : Divise les données en ensembles d'entraînement et de test, normalise les variables numériques, et encode les variables catégorielles et ordinales.

## Architecture des Modules
1. **data_loader.py** : Ingestion des données et sauvegarde du jeu de données transformé pour les étapes ultérieures.
2. **quality_checks.py** : Validation des données avec **Great Expectations**. Vérifie que les types de données et les valeurs des colonnes `age` et `sex` sont conformes aux attentes.
3. **preprocessing.py** : Prétraitement des données en utilisant des pipelines de transformation (imputation, normalisation, encodage).
4. **server.py** : Serveur FastAPI qui expose les endpoints HTTP pour interagir avec le pipeline.

## Organisation des Données et Sortie
- **Data Loader** : Charge et sauvegarde les données dans `data/processed_heart.csv`.
- **Contrôles de Qualité** : Enregistre les résultats des validations pour chaque colonne, permettant un diagnostic rapide des problèmes de conformité.
- **Prétraitement** : Produit les fichiers de données prétraitées dans le répertoire `data/` :
   - `X_train_clean.csv`
   - `X_test_clean.csv`
   - `y_train.csv`
   - `y_test.csv`

---

## Diagramme de Flux des Données

1. **Ingestion** → 2. **Contrôles de Qualité** → 3. **Prétraitement**

1. **Ingestion** :
   - Lit les données brutes depuis `data/heart.csv`.
   - Sauvegarde le fichier transformé en tant que `data/processed_heart.csv`.

2. **Contrôles de Qualité** :
   - Utilise **Great Expectations** pour vérifier la conformité des données selon la configuration dans `config.yaml`.
   - Effectue les validations sur les types de données et les valeurs spécifiques.

3. **Prétraitement** :
   - Divise les données en ensembles d'entraînement et de test.
   - Exécute des transformations comme la normalisation et l'encodage.
   - Sauvegarde les ensembles d'entraînement et de test prêts pour la modélisation.

---

## Technologies Utilisées
- **Pandas** : Pour la manipulation des données.
- **Great Expectations** : Pour les contrôles de qualité avancés.
- **FastAPI** : Pour l'exposition des endpoints API.
- **scikit-learn** : Pour le prétraitement des données.



---

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
│   ├── deploy_pipeline_prod.py    # Utilisez pour le modèle déployer (Sauvegarde du preprocesseur et du modèle en deux fichiers pickle séparé)
│   ├── deploy_server.py    # Lance un conteneur docker pour l'inférence
│   ├── make_predictions.py    # Test du déploiement avec des données d'exemple
│   ├── validate_model.py    # Test du modèle dans le registry de MLflow
│   ├── pipeline_logistic_regression_one_flow.py        # Pipeline avec régression logistique; comprend un pipeline qui associe le preprocessing au modèle
│   ├── pipeline_logisticregression_task_oriented.py        # Pipeline avec régression logistique
│   ├── pipeline_randomforest_task_oriented.py        # Pipeline avec RandomForest
│   ├── pipeline_random_forest__one_flow.py        # Pipeline avec RandomForest;  comprend un pipeline qui associe le preprocessing au modèle
│   ├── data_loader.py       # Chargement des données
│   ├── preprocessing.py     # Prétraitement des données
│   ├── quality_checks.py    # Contrôle de la qualité des données
│   └── server.py            # Serveur pour exécution des workflows
├── /artifacts               # Artefacts générés : graphes, métriques, modèles
├── /great_expectations      # Vérification de la qualité des données (expectations)
└── config.yaml              # Fichier de configuration pour l'orchestration

```
# Script de Déploiement - `deploy_server.py`

Ce script est utilisé pour déployer un modèle MLflow entraîné en tant que conteneur Docker, qui sert le modèle pour l'inférence. Assurez-vous que les serveurs Prefect et MLflow sont en ligne et accessibles avant d'exécuter ce script. De plus, ce script nécessite un environnement virtuel Poetry actif avec toutes les dépendances installées.

## Prérequis

1. **Serveurs Prefect et MLflow** : Assurez-vous que les serveurs Prefect et MLflow sont en cours d'exécution et accessibles sur votre réseau.

2. **Environnement virtuel Poetry** :
   - Assurez-vous d'avoir installé [Poetry](https://python-poetry.org/) pour la gestion des dépendances et de l'environnement virtuel.
   - Activez l'environnement virtuel avec :
     ```bash
     poetry shell
     ```

3. **Dépendances** : Toutes les dépendances doivent être installées dans l'environnement Poetry. Si ce n'est pas le cas, installez-les avec :
   ```bash
   poetry install
   python scripts/deploy_server.py

Le script va :

Configurer le MLFLOW_TRACKING_URI approprié en fonction de votre système d'exploitation.
Construire une image Docker pour le modèle en utilisant l'URI du modèle MLflow.
Déployer le conteneur Docker avec le serveur de modèle exposé sur le port 5001.

### Remarques
Assurez-vous que Docker est installé et en cours d'exécution, car le script utilise les commandes Docker pour construire et exécuter le conteneur du modèle.
Vérifiez aussi que les serveurs Prefect et MLflow sont accessibles.


# Test Script - `make_predictions.py`
# Projet de Prédiction de Maladies Cardiaques avec MLflow et Prefect

Ce script utilise un modèle de classification pour prédire les maladies cardiaques à partir de données cliniques, en s'appuyant sur un pipeline de traitement de données et d'entraînement de modèle, ainsi que sur un serveur de prédiction. Le modèle et le préprocesseur sont enregistrés dans MLflow et peuvent être utilisés pour des inférences sur des serveurs déployés.

## Prérequis

Avant de lancer le script, assurez-vous d'avoir les éléments suivants :
- **MLflow** configuré et en cours d'exécution, avec l'URI de suivi pointant vers `http://localhost:5000`.
- **Prefect** pour orchestrer le pipeline de données `http://localhost:4200`.
- **Python** 3.11.1  avec les librairies suivantes :
  - `mlflow`
  - `cloudpickle`
  - `pandas`
  - `scikit-learn`
  - `seaborn`
  - `matplotlib`
  - `requests`

Ces dépendances seront déjà présent si vous activer le shell poetry et vous éxécuter la commande `poetry install`	

Exécutez :
```shell poetry ```

`python scripts/make_predictions.py`
