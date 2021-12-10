# Libraries for modelling
from fastai import *
from fastai.tabular.all import *
from torch.nn import Sigmoid

from scripts.metrics import (
    macro_averaged_mean_absolute_error,
    macro_averaged_mean_squared_error
)
from scripts.data_preparation import mock_multi_label, merge_station_data

from sklearn.metrics import classification_report


def get_preds(arr):
    """Extract predictions from learner output."""
    mask = arr == 0
    return np.where(mask.any(axis=1), mask.argmax(axis=1), 4)


def replace_actfn(model, actfn):
    """
    Change the activation function in all layers of the NN.
    
    User utkb. Retrieved the 15 Dec 2020. 
    Avaliable at: https://forums.fast.ai/t/change-activation-function-in-resnet-model/78456/12
    """
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, actfn)
        else:
            replace_actfn(child, actfn)


def fit_and_predict(data_subset: typing.Optional[str]=None,
                    engineer_date: bool=True,
                    test_split: str = '2015-01-01',
                    metrics = [F1ScoreMulti(average='weighted')],
                    procs = [Normalize, Categorify],
                    verbose=True, 
                    **learner_kwargs):
    """Implements ordinal regression as multi-label classification with NN."""

    if engineer_date and Categorify not in procs:
        raise Exception("Categorify proc must be in procs if engineer_date is True.")
    # Preprocess and merge station data
    df = merge_station_data(
        engineer_date=engineer_date,
        verbose=verbose
    )
    if data_subset:
        mask_subset = df.date < '2015-01-01'
        df = df[mask_subset]
    # Setup splitting for fast.ai
    mask = df.date >= test_split
    val_idx = df.index[mask]
    splitter = IndexSplitter(val_idx)
    df = df.drop("date", axis=1)
    
    # If we want to engineer dates, add additional cat. vars.
    date_cols = None
    if engineer_date: 
        date_cols = ["Year", "Month", "Week", "Day", "Dayofweek",
                     "Dayofyear", "Is_month_end", "Is_month_start",
                     "Is_quarter_end", "Is_quarter_start", "Is_year_end", 
                     "Is_year_start"]
    ncon_cols = date_cols + ["dangerLevel", "station"] if date_cols else ["dangerLevel", "station"]
    cont_cols = list(df.columns[~df.columns.isin(ncon_cols)])

    # Setup multilabel version of labels
    df_r = mock_multi_label(df).drop("dangerLevel", axis=1)

    # Build fast.ai tabular_learner
    y_names=["dl__1", "dl__2", "dl__3", "dl__4"]
    to = TabularPandas(
        df=df_r,
        cont_names = list(cont_cols),
        cat_names = date_cols + ['station'] if engineer_date else ['station'],
        procs=procs,
        y_names=y_names,
        y_block=MultiCategoryBlock(encoded=True, vocab=y_names),
        splits=splitter(range_of(df_r))
    )
    ## Get batch size, epochs, and activation function
    bs = learner_kwargs.pop("bs")
    epochs = learner_kwargs.pop("epochs")
    actfn = learner_kwargs.pop("actfn")
    
    dls = to.dataloaders(bs=bs)
    ## Build learner
    learn = tabular_learner(dls, metrics=metrics, **learner_kwargs)
    replace_actfn(learn.model, actfn)
    
    # Optionally, plot learning rate
    if verbose:
        learn.lr_find()

    # Fit to training data
    if not verbose:
        with learn.no_logging():
            learn.fit_one_cycle(epochs)
    else:
        learn.fit_one_cycle(epochs)

    # Predict on test/validation set
    df_test = df[mask]
    y_true = df_test['dangerLevel']
    print(len(y_true))
    test_features = df_test.drop(['dangerLevel'], axis=1)
    dl = learn.dls.test_dl(test_features)
    p = learn.get_preds(dl=dl)
    p_bool = (p[0] > 0.5).cpu().numpy()
    y_pred = get_preds(p_bool)
    
    if verbose:
        # Report performance
        print(classification_report(y_true=y_true, y_pred=y_pred))
        print("\nMacro-averaged mean squared error: {macmse}".format(
            macmse=macro_averaged_mean_squared_error(
                y_true=y_true, 
                y_pred=y_pred,
                average=None
            )
        ))
        print("\nMacro-averaged mean absolute error: {macmad}".format(
            macmad=macro_averaged_mean_absolute_error(
                 y_true=y_true, 
                 y_pred=y_pred,
                 average=None
            )
        ))

    return y_pred, y_true, learn