import pandas as pd
import numpy as np
import json
from pickle5 import pickle
from flask import request, Flask,jsonify
from stringmatchmodel import cosinesim, fitmdl

app =Flask(__name__)


@app.route('/train/format/learn', methods=['POST'])

def unsupervisedLearning():
    importData = request.get_json()
    trainData = pd.DataFrame(importData['mappings'])
    trainData['confidence'] = [cosinesim(x, y) for x, y in
                               zip(trainData['sourceField'], trainData['targetField'])]
    clf = fitmdl(trainData['sourceField'], trainData['targetField'])
    pickle.dump(clf, open('text_learning.pickle', 'wb'))
    return json.dumps({"sourceformatName": importData['source'].get('formatName'),
                       "targetformatName": importData['target'].get('formatName'),
                       "overallConfidence": np.mean(trainData['confidence']),
                       "Message": "Learned the mappings"})


if __name__ == '__main__':
    app.run(debug=True, port=5001)
