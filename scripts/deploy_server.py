import os
import subprocess

# Set the MLFLOW_TRACKING_URI based on the OS
tracking_uri = "http://localhost:5000"
if os.name == "nt":  # Windows
    # For Windows CMD
    subprocess.run(["set", f"MLFLOW_TRACKING_URI={tracking_uri}"], shell=True)
    # For Windows PowerShell
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
else:  # Linux or MacOS
    # For Unix-based systems
    subprocess.run(["export", f"MLFLOW_TRACKING_URI={tracking_uri}"], shell=True)
    # os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

# URI of the MLflow model
logged_model = "runs:/a9f6bf7101724ab591340cb90b5aff2e/artifacts"

# Build the Docker image for the model
subprocess.run([
    "mlflow", "models", "build-docker",
    "-m", logged_model,
    "-n", "mlmodelprod",
    "--enable-mlserver"
])

# Run the Docker container with the model served at port 5001
subprocess.run([
    "docker", "run",
    "-p", "5001:8080",
    "mlmodelprod",
    "-t", "production"
])
