import itertools
import pandas as pd

from scripts.data_preparation import merge_station_data 
from scripts.models.cheng_model import fit_and_predict
from scripts.metrics import (
    macro_averaged_mean_squared_error,
    macro_averaged_mean_absolute_error
)

from fastai.tabular.all import * 

from torch.nn import (
    ReLU,
    LeakyReLU,
    Sigmoid
)

from tqdm import tqdm
from sklearn.metrics import (
    precision_score,
    recall_score
)

scores = []
recalls = []
precisions = []

for _ in tqdm(range(1000)):
    ## Train with slightly less epochs, to not overfit
    y_pred, y_true, _ = fit_and_predict(**{
        'layers': [100, 100],
        'lr': 0.01,
        'actfn': LeakyReLU(negative_slope=0.01),
        'bs': 64,
        'epochs': 7,
        'engineer_date': True,
    }, verbose=False)
    
    mamae = macro_averaged_mean_absolute_error(y_true, y_pred)
    mamse = macro_averaged_mean_squared_error(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)

    scores.append({"mamae": mamae,"mamse": mamse}) 
    recalls.append(recall)
    precisions.append(precision)
        
scores_df = pd.DataFrame(scores)
recalls_df = pd.DataFrame(recalls)
precisions_df = pd.DataFrame(precisions)

fname = f"./cross_validation/cheng/"

pickle.dump(scores_df.iloc[2:, :], open(fname+"scores_df.p", "wb"))
pickle.dump(recalls_df.iloc[2:, :], open(fname+"recalls_df.p", "wb"))
pickle.dump(precisions_df.iloc[2:, :], open(fname+"precisions_df.p", "wb"))
