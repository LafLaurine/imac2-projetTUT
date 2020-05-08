import os
import redis
import json
import marshal
from flask import Flask
from classifiers.MesoNet import classify

app = Flask(__name__)
cache = redis.Redis(host='redis', port=6379)

batch_size_default = 4

@app.route('/mesonet_analyse')
def mesonet_analyse():
    name_classifier = os.getenv("mesonet_classifier")
    dir_input = os.getenv("path_to_dataset")
    batch_size = int(os.getenv("batch_size"))
    prediction = classify.analyse_from_dir(
        name_classifier=name_classifier,
        dir_input=dir_input,
        batch_size=batch_size)
    return json.dumps({"analyse" : prediction.__dict__})