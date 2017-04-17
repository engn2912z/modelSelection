import requests
import pickle
url='http://rashidzia.pythonanywhere.com/'


def requestDataset(regression_id=0):
    """downloads a regression dataset from course pythonanywhere site

    Parameters
    ----------
    regression_id : int (optional)
        id number of specific regression dataset

    Returns
    -------
    dataset : dict
        contains numpy arrays (X, y, Xtest) together with regression_id string
    """
    params={}
    if regression_id > 0:
        params['id']=regression_id
    response=requests.get(url+'get_regression',stream=True,params=params)
    dataset=pickle.loads(response.content)
    dataset['regression_id']=response.headers['regression_id']
    with open('dataset.pickle','wb') as f2: f2.write(pickle.dumps(dataset,protocol=3))
    return dataset

def reloadDataset():
    """opens saved dataset.pickle file if it exists, else downloads new dataset

    Returns
    -------
    dataset : dict
        contains numpy arrays (X, y, Xtest) together with regression_id string
    """
    try:
        with open('dataset.pickle','rb') as f1: dataset=pickle.load('dataset.pickle')
    except:
        dataset=requestDataset()
    return dataset

def submitForGrading(fitPipelineEstimator,dataset,email):
    """submits an sklearn pipeline fit and model for evaluation on course site

    Parameters
    ----------
    fitPipelineEstimator : sklearn.estimator
        fit pipeline estimator

    Returns
    -------
    submissionResponse : string
        response from pythonanywhere server about model evaluation results
    """
    submission={'regression_id':dataset['regression_id'],
        'yPredictions':fitPipelineEstimator.predict(dataset['Xtest']),
        'model_parameters':str(fitPipelineEstimator.get_params(deep=True)),
        'named_steps':str(list(fitPipelineEstimator.named_steps.keys())),
        'email':email}
    submissionPickle=pickle.dumps(submission,protocol=3)
    files={'file':('submission.pickle',submissionPickle, 'application/octet-stream')}
    submissionResponse=requests.post(url+'score_regression',files=files)
    return submissionResponse.text
