import pandas as pd
import numpy as np
from scipy import signal
from tqdm.auto import tqdm

tqdm.pandas()   # registers .progress_apply with pandas

cleaned_file = "data/processed/2_cleaned_cgm.csv"  # Path to save the cleaned file
df = pd.read_csv(cleaned_file)  # Load the cleaned data
df['time'] = pd.to_datetime(df['time'])
df.sort_values(['ID', 'time'], inplace=True)
df.set_index('time', inplace=True)

# Select only the relevant data for analysis
df_dexcom = df[df['device']== 'intervals_5mins']
df_dexcom = df_dexcom[~((df_dexcom['ID'].str.startswith('dexi'))|(df_dexcom['ID'].str.startswith('dexip')))]


# ----- helpers ------------------------------------------------------------- #
def bgi_mgdl(g):
    # Kovatchev 2006 constants for mg/dL
    return 1.509 * (np.log(g) ** 1.084 - 5.381)

def mage_window(x):
    if x.size < 2:          # need at least two points to compute a diff
        return np.nan
    sd = np.std(x)
    peaks, _   = signal.find_peaks(x,  prominence=sd)
    troughs, _ = signal.find_peaks(-x, prominence=sd)
    pts   = np.sort(np.concatenate((peaks, troughs, [0, x.size - 1])))
    diffs = np.abs(np.diff(x[pts]))
    return diffs.mean() if diffs.size else np.nan

# ----- main rolling summary ------------------------------------------------ #
def calculate_metrics(group):
    # ── add the 20 min‐ago glucose ─────────────────────────────────────────────
    group["glc_20_min_ago"] = group["glc"].shift(4)
    
    rolled = group["glc"].rolling("1h")      # no min_periods → every row gets output

    # how many CGM points were in that hour?
    group["samples_1h"] = rolled.count()

    # core summaries
    group["avg_glucose"] = rolled.mean()
    group["sd_glucose"]  = rolled.std()

    # time in range (mg/dL)
    group["time_below_70"]  = rolled.apply(lambda x: (x <  70).mean(), raw=False)
    group["time_70_180"]    = rolled.apply(lambda x: ((x >= 70) & (x <= 180)).mean(), raw=False)
    group["time_above_180"] = rolled.apply(lambda x: (x > 180).mean(),             raw=False)

    # glycaemic risk indices
    group["hbgi"] = rolled.apply(lambda x: 10 * np.mean(np.square(np.maximum(bgi_mgdl(x), 0))),
                                 raw=False)
    group["lbgi"] = rolled.apply(lambda x: 10 * np.mean(np.square(np.minimum(bgi_mgdl(x), 0))),
                                 raw=False)

    # MAGE (1-h window)
    group["mage"] = rolled.apply(mage_window, raw=True)

    return group

from tsfresh.feature_extraction import EfficientFCParameters

# The full default set is 700+ features – far too slow for every 5-min sample.
# EfficientFCParameters gives ~60 low-cost ones.
ts_cfg = EfficientFCParameters()

# If you only want a handful (e.g., autocorr lag-1, skewness, kurtosis) you can
# pass a dict like {"absolute_sum_of_changes": None, "autocorrelation": [{"lag": 1}]}

def build_windows_for_tsfresh(group):
    """
    group: one participant's CGM, indexed by time, columns ['glc', ...]
    returns: DataFrame with cols ['id', 'time', 'value'] ready for tsfresh
    """
    # Create a unique window id for each row = current timestamp
    idx = group.index                      # DateTimeIndex
    window_ids = np.arange(len(idx))       # or use idx.astype("int64")

    # Freeze the glucose slice for each row's PRECEDING hour.
    # We collect them in one list to avoid Python loops.
    series_list = []
    for wid, t_end in zip(window_ids, idx):
        win = group.loc[(t_end - pd.Timedelta("1h")): t_end, "glc"]
        if win.empty:
            continue
        tmp = pd.DataFrame(
            {"id": wid,
             "time": win.index.astype("int64"),   # int time is fine for tsfresh
             "value": win.values}
        )
        series_list.append(tmp)

    return pd.concat(series_list, ignore_index=True)

from tsfresh import extract_features

def tsfresh_features_for_participant(group):
    # build the long table
    long_df = build_windows_for_tsfresh(group)

    # run tsfresh ONCE
    feats = extract_features(
        long_df,
        column_id="id",
        column_sort="time",
        column_kind=None,
        column_value="value",
        default_fc_parameters=ts_cfg,
        n_jobs=0,                      # 0 → use all cores once, not per row
        disable_progressbar=True,
    )

    # feats index is window id; align back to the original timestamps
    feats.index = group.index[:len(feats)]   # same order we created ids

    # concatenate with the simple rolling metrics you already have
    out = pd.concat([group.reset_index(), feats.reset_index(drop=True)], axis=1)
    return out

metrics_df = (
    df_dexcom
      .sort_values(["ID", "time"])
      .groupby("ID", group_keys=False)
      .progress_apply(calculate_metrics)          # ← progress bar here
      .groupby("ID", group_keys=False)
      .progress_apply(tsfresh_features_for_participant)  # ← and here
      .reset_index()                               # bring 'time' back
)

metrics_df.to_csv("data/processed/3_features_cgm.csv", index=False)
