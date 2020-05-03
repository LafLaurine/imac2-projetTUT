import os
import redis
import json
from flask import Flask
from classifiers.MesoNet import classify

app = Flask(__name__)
cache = redis.Redis(host='redis', port=6379)

classifier_name_default = 'MESO4_DF'
step_save_weights_temp_default = 5

@app.route('/mesonet_training')
def mesonet_train():
    name_classifier = os.getenv("mesonet_classifier")
    dir_dataset = os.getenv("path_to_dataset")
    batch_size = int(os.getenv("batch_size"))
    number_epochs = int(os.getenv("number_epochs"))
    step_save_weights_temp = int(os.getenv("step_save_checkpoint"))
    evals_learning = classify.learn_from_dir(
        name_classifier=name_classifier,
        dir_dataset=dir_dataset,
        batch_size=batch_size,
        number_epochs=number_epochs,
        step_save_weights_temp=step_save_weights_temp)
    return json.dumps(str(evals_learning.__dict__))