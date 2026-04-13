#!/usr/bin/env python3
"""Minimal prepare with spatiotemporal absence sampling and intersection of dynamic dates.
   Lightning NaNs are treated as 0 (no lightning)."""

from pathlib import Path
import geopandas as gpd
import xarray as xr
import numpy as np
import rioxarray
import pandas as pd
from tqdm import tqdm
from scipy.spatial import cKDTree

# ========== CONFIG ==========
HAZARD_TYPE = "landslide"
WINDOW_DAYS = list(range(1, 61))
VAR_SPECS = {
    "precipitation": ("Precipitation", "prec_*.nc", "prec", "sum"),
    "temperature":   ("Temperature",   "tmean_*.nc", "tmean", "mean"),
    "lightning":     ("Lightning_Standardized", "lightning_density_*.nc", "lightning_density", "max"),
}

ROOT = Path(__file__).resolve().parent / "autosearch_data"
EVENT_FILE = ROOT / f"hazards_{HAZARD_TYPE}s.gpkg"
STATIC_DIR = ROOT / "static"
DYNAMIC_DIR = ROOT / "dynamic"
OUTPUT_FILE = ROOT / f"prepared_clean_{HAZARD_TYPE}_ebm.npz"

SLOPE_TIF = STATIC_DIR / "slope.tif"
SLOPE_THRESH = 5.0
TEMPORAL_BUFFER = 120          # days
MIN_DIST_EVENT = 500.0         # meters
ABSENCE_RATIO = 5
BIAS_RASTER = "bias_sampling.tif"
# ============================

def extract_vals(raster_path, x, y):
    da = rioxarray.open_rasterio(raster_path).squeeze()
    if 'band' in da.dims:
        da = da.isel(band=0)
    return da.sel(x=xr.DataArray(x), y=xr.DataArray(y), method='nearest').values.astype(np.float32)

def get_common_valid_days(t0):
    """Return a set of day offsets (relative to t0) that exist in ALL dynamic NetCDF files."""
    common_days = None
    for var_name, (subdir, pat, nc_var, agg) in VAR_SPECS.items():
        files = sorted((DYNAMIC_DIR / subdir).glob(pat))
        if not files:
            print(f"Warning: No files for {var_name}, skipping.")
            continue
        ds = xr.open_mfdataset(files, combine='nested', concat_dim='time')
        times = pd.to_datetime(ds.time.values).floor('D')
        ds.close()
        days = set((times - t0).days)
        if common_days is None:
            common_days = days
        else:
            common_days &= days
        if not common_days:
            raise RuntimeError(f"No common days left after processing {var_name}!")
    if common_days is None:
        raise RuntimeError("No dynamic variables found!")
    return common_days

def main():
    rng = np.random.default_rng(42)
    events = gpd.read_file(EVENT_FILE)
    events["date"] = pd.to_datetime(events["date"]).dt.floor('D')
    t0 = events["date"].min()
    event_days = (events["date"] - t0).dt.days.values.astype(np.int32)
    ev_x, ev_y = events.geometry.x.values, events.geometry.y.values
    ev_coords = np.column_stack([ev_x, ev_y])

    all_dates = pd.date_range(t0, events["date"].max(), freq='D')
    day_index = {d: i for i, d in enumerate((all_dates - t0).days)}
    all_days = np.array(list(day_index.values()))

    # Trivial terrain mask (slope >= 5°)
    slope_da = rioxarray.open_rasterio(SLOPE_TIF).squeeze()
    valid_mask = (slope_da >= SLOPE_THRESH).astype(np.uint8)
    valid_y, valid_x = np.where(valid_mask.values == 1)
    valid_cells = np.column_stack([valid_mask.x.values[valid_x], valid_mask.y.values[valid_y]])

    # Absence sampling
    n_abs = len(events) * ABSENCE_RATIO
    n_same = n_abs // 2
    n_diff = n_abs - n_same

    # (a) Same location, different time (120‑day buffer)
    same_data = []
    for i in range(len(events)):
        ev_day = event_days[i]
        possible = all_days[np.abs(all_days - ev_day) > TEMPORAL_BUFFER]
        if len(possible) == 0:
            continue
        for _ in range(max(1, n_same // len(events))):
            day = rng.choice(possible)
            same_data.append((ev_x[i], ev_y[i], day))
    same_df = pd.DataFrame(same_data, columns=['x','y','day']).sample(n=min(n_same, len(same_data)), random_state=42)

    # (b) Different location (≥500 m from events), random day
    tree = cKDTree(ev_coords)
    diff_data = []
    while len(diff_data) < n_diff:
        idx = rng.choice(len(valid_cells), size=n_diff*2, replace=True)
        cand = valid_cells[idx]
        dist, _ = tree.query(cand)
        keep = dist >= MIN_DIST_EVENT
        for xy, d in zip(cand[keep], rng.choice(all_days, size=keep.sum(), replace=True)):
            diff_data.append((xy[0], xy[1], d))
    diff_df = pd.DataFrame(diff_data, columns=['x','y','day']).iloc[:n_diff]

    absence_df = pd.concat([same_df, diff_df], ignore_index=True)
    absence_df['is_event'] = 0
    events_df = pd.DataFrame({'x': ev_x, 'y': ev_y, 'day': event_days, 'is_event': 1})
    all_samples = pd.concat([events_df, absence_df], ignore_index=True)

    # Monthly balancing (absences only)
    all_samples['date'] = t0 + pd.to_timedelta(all_samples['day'], unit='D')
    all_samples['month'] = all_samples['date'].dt.month
    abs_idx = all_samples['is_event'] == 0
    target_per_month = int(np.ceil(len(absence_df) / 12))
    keep_idx = list(all_samples[~abs_idx].index)
    for m in range(1, 13):
        avail = all_samples.index[abs_idx & (all_samples['month'] == m)].tolist()
        chosen = rng.choice(avail, size=min(target_per_month, len(avail)), replace=False)
        keep_idx.extend(chosen)
    all_samples = all_samples.loc[keep_idx].reset_index(drop=True)

    # ---------- Filter to common days across all dynamic NetCDF files ----------
    print("Checking NetCDF date ranges...")
    common_days = get_common_valid_days(t0)
    print(f"  Common valid days across all dynamic variables: {len(common_days)}")

    valid_temporal = all_samples['day'].isin(common_days)
    n_before = len(all_samples)
    all_samples = all_samples[valid_temporal].reset_index(drop=True)
    n_after = len(all_samples)
    print(f"  Retained {n_after} of {n_before} samples within common NetCDF days.")
    n_events_kept = (all_samples['is_event'] == 1).sum()
    print(f"  Events kept: {n_events_kept} of {len(events)}")

    if n_after == 0:
        raise RuntimeError("No samples left after temporal filtering!")

    # Extract static features
    static_paths = sorted(STATIC_DIR.glob("*.tif"))
    bias_path = STATIC_DIR / BIAS_RASTER
    other_static = [p for p in static_paths if p != bias_path]
    static_names = [p.stem for p in other_static]
    static_vals = np.column_stack([extract_vals(p, all_samples.x, all_samples.y) for p in other_static])
    bias_vals = extract_vals(bias_path, all_samples.x, all_samples.y)

    # Extract dynamic features
    n_temp = len(VAR_SPECS) * len(WINDOW_DAYS)
    dynamic = np.zeros((len(all_samples), n_temp), dtype=np.float32)
    feat_idx = 0
    temporal_series = {}
    for var_name, (subdir, pat, nc_var, agg) in VAR_SPECS.items():
        files = sorted((DYNAMIC_DIR / subdir).glob(pat))
        ds = xr.open_mfdataset(files, combine='nested', concat_dim='time')
        da = ds[nc_var]
        times = pd.to_datetime(da.time.values).floor('D')
        day_to_idx = {d: i for i, d in enumerate((times - t0).days.values)}
        temporal_series[f"{var_name}_mean"] = da.mean(dim=['x','y']).values.astype(np.float32)

        ts = da.sel(x=xr.DataArray(all_samples.x), y=xr.DataArray(all_samples.y), method='nearest').compute()
        # Replace NaN with 0 for lightning (no lightning = 0)
        if var_name == "lightning":
            ts = ts.fillna(0.0)
        ts_xr = xr.DataArray(ts, dims=('time','sample'), coords={'time': times})
        for w in tqdm(WINDOW_DAYS, desc=f"  {var_name}"):
            rolled = getattr(ts_xr.rolling(time=w, min_periods=1), agg)()
            for i, row in all_samples.iterrows():
                t_idx = day_to_idx.get(row['day'], None)
                if t_idx is not None:
                    dynamic[i, feat_idx] = rolled.values[t_idx, i]
                # else remains 0 (will be handled later if NaN appears)
            static_names.append(f"{var_name}_{w}d_{agg}")
            feat_idx += 1
        ds.close()

    # Check for NaNs in dynamic (should only come from non-lightning variables or spatial gaps)
    nan_rows = np.isnan(dynamic).any(axis=1) | np.isnan(static_vals).any(axis=1)
    n_nan = nan_rows.sum()
    if n_nan > 0:
        print(f"  Dropping {n_nan} samples with NaN values (likely spatial data gaps in precip/temp).")
        all_samples = all_samples[~nan_rows].reset_index(drop=True)
        static_vals = static_vals[~nan_rows]
        bias_vals = bias_vals[~nan_rows]
        dynamic = dynamic[~nan_rows]

    X = np.hstack([static_vals, dynamic])
    y = all_samples.is_event.values.astype(np.int8)
    event_mask = y == 1

    # Pixel area for weights
    with rioxarray.open_rasterio(other_static[0]) as da:
        res = da.rio.resolution()
        pixel_area = abs(res[0] * res[1])

    np.savez_compressed(
        OUTPUT_FILE,
        t0=str(t0),
        feature_names=static_names,
        X=X, y=y,
        event_X=X[event_mask],
        event_coords=all_samples[event_mask][['x','y']].values,
        event_days=all_samples[event_mask]['day'].values,
        event_fold=np.zeros(event_mask.sum(), dtype=np.int16),  # placeholder
        event_bias=bias_vals[event_mask],
        support_X=X[~event_mask],
        support_bias=bias_vals[~event_mask],
        weights=np.full(len(all_samples), pixel_area, dtype=np.float32),
        **temporal_series
    )
    print(f"Saved {OUTPUT_FILE}")

if __name__ == "__main__":
    main()