from flask import Flask
from flask_restful import Resource, Api, reqparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests

checkpoint = 'E:/zz2953/Dorothy/models/fine_tuned_distilbert_sample_10k'
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

class_names = {0: "Democratic", 1: "Republican"}


# Turn into readable class names
def logits_to_class_names(predictions):
        predictions = torch.nn.Softmax(predictions.logits)
        predictions = torch.argmax(predictions).numpy()
        predictions = [class_names[prediction] for prediction in predictions]

        return predictions


# Setting up API
app = Flask(import_name=__name__)
api = Api(app=app)

parser = reqparse.RequestParser()
parser.add_argument(name="Questions", type=str, action="append",
                    help="The question to be classified", required=True)


# Creating a class to represent our endpoint
class Inference(Resource):
    # A method corresponding to a GET request
    def get(self):
        # Parsing the arguments we defined earlier
        args = parser.parse_args()

        # Tokenizing the question
        question = tokenizer(args["Questions"], return_tensors="pt", padding=True)

        # Obtaining a prediction
        prediction = logits_to_class_names(model(question))

        # Returning the prediction
        return {"Predictions": prediction}, 200


# Adding the endpoint to our app
api.add_resource(Inference, "/inference")

# requests stuff
url = "http://127.0.0.1:5000/inference"
data = {"Questions": "Do you support abortion?"}  # Sample question


def query(url, payload):
    return requests.get(url, json=payload)


# Sending a GET request and obtaining the results
response = query(url, data)

# Inspecting the response
print(response.json())

if __name__ == "__main__":
    app.run()  # only use run() for testing, check options for deployment



