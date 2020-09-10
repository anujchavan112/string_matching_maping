import pandas as pd
import numpy as np
from pickle5 import pickle
import json
from flask import request, Flask
from stringmatchmodel import cosinesim, fitmdl

app = Flask(__name__)
trainData = pd.DataFrame()
sourceData = []

@app.route('/train/format/match', methods=['POST'])
def SupervisedLearning():
    importData=request.get_json()
    traindata=pd.DataFrame(importData['target']['formatFields'],columns=['targetField'])
    sourcedata = importData['source']['formatFields']
    sourceFormatName = importData['source'].get('formatName')
    targetformatName = importData['target'].get('formatName')

    traindata['sourceField'], traindata['confidence'] = zip(
        *traindata['targetField'].apply(lambda x: cosinesim(x, sourcedata)))
    traindata = traindata[['sourceField', 'targetField', 'confidence']]

    global trainData
    if trainData.empty:
        trainData=traindata
    else:
        trainData=trainData.append(traindata)

    fitmodel=fitmdl(trainData['sourceField'], trainData['targetField'])
    clf=fitmodel.model()
    pickle.dump({'model': clf, 'labelEncoder': fitmodel.Encode, 'count_vect_fit': fitmodel.x_train_counts,
                 'tfidf_fit': fitmodel.x_train_tfidf}, open('text_matching.pickle', 'wb'))
    return json.dumps({"sourceformatName": sourceFormatName,
                       "targetformatName": targetformatName,
                       "overallConfidence": np.mean(traindata['confidence']),
                       "mappings": traindata.to_dict(orient='records')})


if __name__ == '__main__':
    app.run(debug=True, port=5002)
