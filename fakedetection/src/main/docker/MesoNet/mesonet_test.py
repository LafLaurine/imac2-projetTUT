import os
import redis
import json
import numpy
from json import JSONEncoder
from flask import Flask
from classifiers.MesoNet import classify


app = Flask(__name__)
cache = redis.Redis(host='redis', port=6379)


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

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
    numpyData = evals_test.__dict__
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
    return encodedNumpyData
