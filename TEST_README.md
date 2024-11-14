
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