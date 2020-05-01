from classifiers.CapsuleForensics import classify as clf
import redis
import os
import json
from flask import Flask

app = Flask(__name__)
cache = redis.Redis(host='redis', port=6379)

root_checkpoint = 'checkpoints'

@app.route('/capsule_forensics_analyse')

def capsule_forensics_analyse():
    name_classifier = os.getenv("capsule_forensics_classifier")
    dir_input = os.getenv("path_to_dataset")
    batch_size = int(os.getenv("batch_size"))
    version_weights = int(os.getenv("version_weights"))
    
    prediction = clf.analyse_from_dir(
        method_classifier=name_classifier,
        dir_input=dir_input,
        version_weights=version_weights,
        root_checkpoint=root_checkpoint,
        batch_size=batch_size)
    prediction.print()
    s = '{"message" : "CapsuleForensics analyse done" }'
    return json.loads(s)