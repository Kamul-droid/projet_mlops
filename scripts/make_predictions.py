import mlflow

mlflow.set_tracking_uri('http://localhost:5000')
logged_model = 'runs:/e095852ec5f64e89894a61edb466ce8e/artifacts'


# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model( model_uri=logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

# Préparer un exemple de données d'entrée sous forme de DataFrame
data = {
    "age": [55],
    "sex": [1],  
    "cp": [2],
    "trestbps": [130],
    "chol": [250],
    "fbs": [0],
    "restecg": [1],
    "thalach": [160],
    "exang": [0],
    "oldpeak": [2.3],
    "slope": [2],
    "ca": [0],
    "thal": [1]
}
pred =loaded_model.predict(pd.DataFrame(data))

print (pred)


# curl -X POST -H "Content-Type: application/json" \
#   -d '{
#         "columns": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
#         "data": [[55, 1, 2, 130, 250, 0, 1, 160, 0, 2.3, 2, 0, 1]]
#       }' \
#   http://127.0.0.1:5000/invocations


