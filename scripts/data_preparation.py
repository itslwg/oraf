import os
import typing

import pandas as pd
import numpy as np

from fastai.tabular.all import add_datepart

def concatenate_with_shifts(df: pd.DataFrame, periods: int,
                            use_prev_target: bool=True) -> pd.DataFrame:
    """Simply shifts and appends prefix to column names."""
    d = [df.shift(period) for period in range(1, periods + 1)]
    for i, df_ in enumerate(d):
        df_.columns = np.add(df_.columns, "__s" + str(i + 1))
        d[i] = df_
    dfs = [df] + d

    return pd.concat(dfs, axis=1)


def target_as_features(y: pd.DataFrame, lookback: int,
                       merge_high_cat: bool=True) -> pd.DataFrame:
    if merge_high_cat:
        y.loc[:, "dangerLevel"] = y.loc[:, "dangerLevel"].replace(to_replace=5.0, value=4.0)
    days = range(1, lookback + 1)
    previous_targets = pd.concat([y.shift(i) for i in days], axis=1)
    labels = ['dangerLevel__s' + str(i) for i in days]
    previous_targets.columns = labels
    return previous_targets.dropna()


def preprocess_data(df: pd.DataFrame, y_label: str="dangerLevel",
                    periods: int=7, merge_high_cat: bool=True,
                    agg_function: typing.Optional[typing.Callable]=None,
                    engineer_date: bool=False, verbose: bool=True) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    # Descriptory variables
    desc_cols = [
        "warnregionVersion",
        "station_code",
        "sectorId",
        "name",
        "Altitude",
        "validToDate",
        "dangerLevelDetail",
        "aspectFrom",
        "aspectEnd",
        "altitudeLow",
        "altitudeHigh",
        "conditions",
        "hours_valid"
    ]
    # Variables with constant variance, and
    # variables that Michele has advised to drop
    constant_cols = [
        'MS_SN_Runoff',
        'Sclass1', 
        'zS5',
        'Sd',
        'Sn',
        'Ss',
        'S4',
        'S5'
    ]
    drop_cols = desc_cols + constant_cols
    station_code = df.station_code.unique()
    df_c = df.drop(columns=drop_cols)
    df_c = df_c.set_index("measure_date")

    # Missingness handling
    df_m = df_c.dropna(axis=1, how='all')
    new_index = pd.date_range(start=df_m.index[0],end=df_m.index[-1], freq='3H')
    df_f = df_m.reindex(new_index, method='ffill', limit=16).rename_axis("date")

    # Data mendling
    y = df_f[["dangerLevel"]].between_time('18:00', '18:00')
    features = df_f.drop(columns=["dangerLevel", "datum"])
    if agg_function:
        features = features.resample('24h', origin="1997-11-11 18:00", label="right").agg(agg_function, skipna=False)
    else:
        features = concatenate_with_shifts(features, periods = periods).shift(1)
    df_concat_nna = features.dropna()  # Drops rows where shifted period is NA
    
    # Final touches
    previous_targets = target_as_features(y, lookback=2, merge_high_cat=merge_high_cat)
    j = df_concat_nna.join(previous_targets, how='inner').join(y, how='left')
    jd = j.loc[j.dangerLevel.notnull()].reset_index()
    if merge_high_cat:
        n = jd.loc[:, "dangerLevel"]
        n_replaced = n.replace(to_replace=5.0, value=4.0)
        jd["dangerLevel"] = n
    if engineer_date:
        jd = add_datepart(jd, "date", drop=False).drop("Elapsed", axis=1)
    if verbose:
        print("Processed data for station: " + station_code)

    return jd


def merge_station_data(engineer_date: bool, verbose: bool=True, merge_high_cat: bool=True):
    """Load, prepare and merge the data"""
    date_cols = [
        "measure_date",
        "datum"
    ]
    # Get all csv files.
    ds = {
        k.split(".")[0]: pd.read_csv("./data/raw/" + k, 
                                     parse_dates=date_cols).iloc[:, 1:]
        for k in os.listdir("./data/raw") if k.endswith(".csv")
    }
    ks = list(ds.keys())
    # Preprocess 
    dfs = [
        preprocess_data(
            ds[k], 
            engineer_date=engineer_date,
            verbose=verbose,
            merge_high_cat=merge_high_cat
        )
        for k in ks
    ]
    # Merge
    return pd.concat(dfs, keys = ks).reset_index(level = 0).rename(columns=dict(level_0="station"))


def mock_multi_label(df: pd.DataFrame) -> pd.DataFrame:
    """fast.ai helper to create tabular multi-label."""
    one,two,three,four = [],[],[],[]
    labels = pd.Series(list(range(1, 5)))
    for row in df.itertuples(name='Pandas'):
        r = pd.Series(list(range(1, int(row.dangerLevel) + 1)))
        b = labels.isin(r)
        one.append(b[0])
        two.append(b[1])
        three.append(b[2])
        four.append(b[3])
    df['dl__1'] = np.array(one)
    df['dl__2'] = np.array(two)
    df['dl__3'] = np.array(three)
    df['dl__4'] = np.array(four)

    return df