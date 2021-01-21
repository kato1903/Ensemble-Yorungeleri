from scipy.io import arff
from io import StringIO
import numpy as np

def arff_to_numpy(path):
    with open(path) as f:
        content = f.read()
        data, meta = arff.loadarff(StringIO(content))

    dataset = np.array(data.tolist(),dtype=np.float)
    
    X = dataset[:,:-1]
    y = dataset[:,-1]
    y -= 1 
    
    return X,y