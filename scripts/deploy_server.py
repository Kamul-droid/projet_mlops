import subprocess

# URI du modèle MLflow
logged_model = 'runs:/33a1262e152c4c608c074762e0992feb/model'

# Déployer le modèle en utilisant la commande `mlflow models serve`
subprocess.run(["mlflow", "models", "serve", "-m", logged_model, "-p", "3000"])


