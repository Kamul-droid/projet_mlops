from mlflow.models import validate_serving_input

model_uri = 'runs:/e095852ec5f64e89894a61edb466ce8e/artifacts'

# The logged model does not contain an input_example.
# Manually generate a serving payload to verify your model prior to deployment.
from mlflow.models import convert_input_example_to_serving_input

# Define INPUT_EXAMPLE via assignment with your own input example to the model
# A valid input example is a data instance suitable for pyfunc prediction
INPUT_EXAMPLE = {
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
serving_payload = convert_input_example_to_serving_input(INPUT_EXAMPLE)

# Validate the serving payload works on the model
validate_serving_input(model_uri, serving_payload)

print(validate_serving_input)