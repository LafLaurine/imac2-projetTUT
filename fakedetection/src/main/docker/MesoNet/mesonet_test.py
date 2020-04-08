import os
import redis
from flask import Flask
from classifiers.MesoNet import classify


app = Flask(__name__)
cache = redis.Redis(host='redis', port=6379)

@app.route('/mesonet_test')

def mesonet_test():
    name_classifier = os.getenv("mesonet_classifier")
    dir_dataset_test = os.getenv("path_to_dataset")
    batch_size = int(os.getenv("batch_size"))
    number_epochs = int(os.getenv("number_epochs"))
    evals_test = classify.test_from_dir(
        name_classifier=name_classifier,
        dir_dataset_test=dir_dataset_test,
        batch_size=batch_size,
        number_epochs=number_epochs)
    evals_test.print()
    s = '{"message" : "Test mesonet ok"}'
    return json.loads(s)