import os
import redis
import json
import numpy as np
from flask import Flask
from classifiers.MesoNet import classify as clf

app = Flask(__name__)
cache = redis.Redis(host='redis', port=6379)

classifier_name_default = 'MESO4_DF'
step_save_weights_temp_default = 5

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

@app.route('/mesonet_training')
def mesonet_train():
    name_classifier = os.getenv("mesonet_classifier")
    dir_dataset = os.getenv("train_dataset")
    batch_size = int(os.getenv("batch_size"))
    number_epochs = int(os.getenv("number_epochs"))
    step_save_weights_temp = int(os.getenv("step_save_checkpoint"))
    evals_learning = clf.learn_from_dir(
        name_classifier=name_classifier,
        dir_dataset=dir_dataset,
        batch_size=batch_size,
        number_epochs=number_epochs,
        step_save_weights_temp=step_save_weights_temp)
    return json.dumps(evals_learning.__dict__, cls=MyEncoder)
