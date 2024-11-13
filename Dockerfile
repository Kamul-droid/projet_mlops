FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install poetry

RUN poetry install --no-dev

EXPOSE 8000 6200 5000 4201

CMD ["sh", "-c", "\
    poetry run uvicorn scripts.server:app --host 0.0.0.0 --port 8000 & \
    poetry run jupyter lab --port=6200 --ip=0.0.0.0 --no-browser --allow-root & \
    poetry run mlflow ui --backend-store-uri mlflow_run --host 0.0.0.0 --port 5000 & \
    poetry run prefect server start --host 0.0.0.0 --port 4200 & \
    tail -f /dev/null"]