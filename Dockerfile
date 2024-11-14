FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install poetry

RUN poetry install --no-dev

ENV MLFLOW_TRACKING_URI=http://127.0.0.1:5000


EXPOSE 8000 6200 5000 4200

CMD ["sh", "-c", "\
    poetry run uvicorn scripts.server:app --host 0.0.0.0 --port 8000 & \
    poetry run jupyter lab --port=6200 --ip=0.0.0.0 --no-browser --allow-root & \
    poetry run mlflow server --host 0.0.0.0 --port 5000 & \
    poetry run prefect server start --host 0.0.0.0 --port 4200 & \ 
    poetry run python ./scripts/deploy_server.py & \
    tail -f /dev/null"]




# Exposer les ports nécessaires
EXPOSE 8000 6200 5000 4200 3000

# Lancer les différentes applications
CMD ["sh", "-c", "\
    poetry run uvicorn scripts.server:app --host 0.0.0.0 --port 8000 & \
    poetry run jupyter lab --port=6200 --ip=0.0.0.0 --no-browser --allow-root & \
    poetry run mlflow server --host 0.0.0.0 --port 5000 & \
    poetry run prefect server start --host 0.0.0.0 --port 4200 & \
    poetry run python ./scripts/deploy_server.py & \
    tail -f /dev/null"]