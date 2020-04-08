from classifiers.CapsuleForensics import classify as clf
import redis
from flask import Flask

app = Flask(__name__)
cache = redis.Redis(host='redis', port=6379)

@app.route('/capsule_forensics_test')

batch_size_default = 8

root_checkpoint = 'checkpoints'

def capsule_test():
    name_classifier = os.getenv("capsule_forensics_classifier")
    dir_database = os.getenv("path_to_dataset")
    batch_size = int(os.getenv("batch_size"))
    number_epochs =int(os.getenv("number_epochs"))
    version_weights =int(os.getenv("version_weights"))
    evals_test = clf.test_from_dir(
        method_classifier=name_classifier,
        dir_dataset=dir_database,
        version_weights=version_weights,
        root_checkpoint=root_checkpoint,
        batch_size=batch_size,
        number_epochs=number_epochs)
    evals_test.print()
    s = '{"message" : "Working" }'
    return json.loads(s)