# Architecture du Pipeline de Données

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
│   ├── pipeline_2.py        # Pipeline avec régression logistique
│   ├── pipeline_3.py        # Pipeline avec RandomForest
│   ├── data_loader.py       # Chargement des données
│   ├── preprocessing.py     # Prétraitement des données
│   ├── quality_checks.py    # Contrôle de la qualité des données
│   └── server.py            # Serveur pour exécution des workflows
├── /artifacts               # Artefacts générés : graphes, métriques, modèles
├── /great_expectations      # Vérification de la qualité des données (expectations)
└── config.yaml              # Fichier de configuration pour l'orchestration
