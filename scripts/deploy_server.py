import subprocess

import mlflow

# #Windows
# set MLFLOW_TRACKING_URI=http://localhost:5000
# #PS Windows
# $env:MLFLOW_TRACKING_URI="http://localhost:5000"
# #Linux 
# export MLFLOW_TRACKING_URI=http://localhost:5000

# URI du modèle MLflow
logged_model = 'runs:/e095852ec5f64e89894a61edb466ce8e/artifacts'

# Déployer le modèle en utilisant la commande `mlflow models serve`
subprocess.run(["mlflow", "models", "serve", "-m", logged_model, "-p", "3000" ,"--enable-mlserver"])
