
# Deployment Script - `deploy_server.py`

This script is used to deploy a trained MLflow model as a Docker container, which serves the model for inference. Ensure that the Prefect and MLflow servers are online and accessible before running this script. Additionally, this script requires an active Poetry virtual environment with all dependencies installed.

## Prerequisites

1. **Prefect and MLflow Servers**: Ensure both Prefect and MLflow servers are running and accessible on your network.

2. **Poetry Virtual Environment**: 
   - Make sure you have [Poetry](https://python-poetry.org/) installed for managing dependencies and the virtual environment.
   - Activate the virtual environment with:
     ```bash
     poetry shell
     ```

3. **Dependencies**: All dependencies should be installed in the Poetry environment. If not, install them with:
   ```bash
   poetry install
   ```

## Configuration

1. **MLflow Tracking URI**: The script automatically sets the `MLFLOW_TRACKING_URI` environment variable based on your operating system (Windows or Linux).
   
2. **Model URI**: The MLflow model URI is hardcoded in the script as `runs:/a9f6bf7101724ab591340cb90b5aff2e/artifacts`. Update this URI in the script if your model path is different.

## Usage

Once all prerequisites are met, execute the following steps:

1. **Activate the Poetry Virtual Environment**:
   ```bash
   poetry shell
   ```

2. **Run the Deployment Script**:
   ```bash
   python scripts/deploy_server.py
   ```

The script will:
- Set the appropriate `MLFLOW_TRACKING_URI` based on your operating system.
- Build a Docker image for the model using the MLflow model URI.
- Deploy the Docker container with the model server exposed on port 5001.

## Notes

- Ensure Docker is installed and running, as the script uses Docker commands to build and run the model container.
- Confirm that both the Prefect and MLflow servers are accessible and that your MLflow model URI is valid and correctly configured.
