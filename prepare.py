#!/usr/bin/env python3
"""
Hazard-agnostic data preparation and loading.
Run this script once to generate the .npz file.
After that, import it to access data splits for modeling.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
import geopandas as gpd
import xarray as xr
import numpy as np
import rasterio
import pandas as pd

# ============================================================
# CONFIGURATION – EDIT THIS FOR YOUR HAZARD
# ============================================================
ROOT = Path(__file__).resolve().parent

HAZARD_TYPE = "landslide"   # used for file naming

EVENT_FILE = ROOT / f"hazards_{HAZARD_TYPE}s.gpkg"
STATIC_DIR = ROOT / "static"
DYNAMIC_DIR = ROOT / "dynamic"
OUTPUT_FILE = ROOT / f"prepared_point_data_{HAZARD_TYPE}.npz"

QUADRATURE_STRIDE = 12
WINDOW_DAYS = 60

# Bias raster (must exist in STATIC_DIR)
BIAS_RASTER = "bias_sampling.tif"

# Dynamic variables: (subdir, file_pattern, netcdf_var, aggregation)
VAR_SPECS = {
    "precipitation": ("Precipitation", "prec_*.nc", "prec", "sum"),
    "temperature":   ("Temperature", "tmean_*.nc", "tmean", "mean"),
    "lightning":     ("Lightning_Standardized", "lightning_density_*.nc", "lightning_density", "max"),
}
# ============================================================

# ----------------------------------------------------------------------
# Data generation (run once)
# ----------------------------------------------------------------------
def extract_at_points(raster_path, coords):
    with rasterio.open(raster_path) as src:
        rows, cols = rasterio.transform.rowcol(src.transform, coords[:, 0], coords[:, 1])
        return src.read(1)[rows, cols].astype(np.float32)

def generate_data():
    print(f"Generating data for hazard: {HAZARD_TYPE}")
    gdf = gpd.read_file(EVENT_FILE)
    gdf["date"] = pd.to_datetime(gdf["date"]).dt.floor('D')
    event_coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])

    # Spatial fold
    x_mid, y_mid = gdf.geometry.x.median(), gdf.geometry.y.median()
    event_fold = ((gdf.geometry.x >= x_mid).astype(int) + 2*(gdf.geometry.y >= y_mid).astype(int)).astype(np.int16)

    # Support grid from first static raster (non-bias)
    static_paths = sorted(STATIC_DIR.glob("*.tif"))
    bias_path = STATIC_DIR / BIAS_RASTER
    other_static = [p for p in static_paths if p != bias_path]

    with rasterio.open(other_static[0]) as src:
        mask = src.read_masks(1) > 0
        transform, res = src.transform, src.res
        rows, cols = np.where(mask)
        sel = (rows % QUADRATURE_STRIDE == 0) & (cols % QUADRATURE_STRIDE == 0)
        sup_rows, sup_cols = rows[sel], cols[sel]
        xs, ys = rasterio.transform.xy(transform, sup_rows, sup_cols)
        sup_coords = np.column_stack([xs, ys]).astype(np.float32)
        sup_weights = np.full(len(sup_rows), abs(res[0]*res[1]) * QUADRATURE_STRIDE**2, dtype=np.float32)

    # Extract static features
    print("Extracting static features...")
    static_names = [p.stem for p in other_static]
    event_static = np.column_stack([extract_at_points(p, event_coords) for p in other_static])
    sup_static   = np.column_stack([extract_at_points(p, sup_coords)   for p in other_static])
    bias_event   = extract_at_points(bias_path, event_coords)
    bias_sup     = extract_at_points(bias_path, sup_coords)

    # Temporal index
    t0 = gdf["date"].min()
    event_day_idx = (gdf["date"] - t0).dt.days.values
    unique_days = np.unique(event_day_idx)
    temporal_series = {}

    xr_event_x = xr.DataArray(event_coords[:, 0], dims='point')
    xr_event_y = xr.DataArray(event_coords[:, 1], dims='point')
    xr_sup_x   = xr.DataArray(sup_coords[:, 0],   dims='point')
    xr_sup_y   = xr.DataArray(sup_coords[:, 1],   dims='point')

    for var, (subdir, pat, nc_var, agg_func) in VAR_SPECS.items():
        print(f"Processing {var}...")
        ds = xr.open_mfdataset(
            sorted((DYNAMIC_DIR / subdir).glob(pat)),
            combine='nested', concat_dim='time', chunks={'time': -1, 'x': 'auto', 'y': 'auto'}
        )
        da = ds[nc_var].chunk({'time': -1})
        times = pd.to_datetime(da.time.values).floor('D')
        full_day_idx = (times - t0).days.values
        day_to_idx = {d: i for i, d in enumerate(full_day_idx)}

        temporal_series[var] = da.mean(dim=['x', 'y']).values.astype(np.float32)

        event_ts = da.sel(x=xr_event_x, y=xr_event_y, method='nearest').compute()
        sup_ts   = da.sel(x=xr_sup_x,   y=xr_sup_y,   method='nearest').compute()

        event_ts_xr = xr.DataArray(event_ts, dims=('time', 'event'), coords={'time': times})
        sup_ts_xr   = xr.DataArray(sup_ts,   dims=('time', 'support'), coords={'time': times})

        rolled_event = event_ts_xr.rolling(time=WINDOW_DAYS, min_periods=1).reduce(agg_func)
        rolled_sup   = sup_ts_xr.rolling(time=WINDOW_DAYS, min_periods=1).reduce(agg_func)

        event_agg_vals = []
        for day in event_day_idx:
            idx = day_to_idx.get(day - 1)
            if idx is not None:
                event_agg_vals.append(rolled_event.isel(time=idx).values)
            else:
                event_agg_vals.append(np.full(len(event_coords), np.nan))
        sup_agg_vals = []
        for day in unique_days:
            idx = day_to_idx.get(day - 1)
            if idx is not None:
                sup_agg_vals.append(rolled_sup.isel(time=idx).values)
            else:
                sup_agg_vals.append(np.full(len(sup_coords), np.nan))

        event_agg = np.nan_to_num(np.array(event_agg_vals), nan=0.0).astype(np.float32)
        sup_agg   = np.nan_to_num(np.array(sup_agg_vals), nan=0.0).astype(np.float32)

        event_static = np.column_stack([event_static, event_agg])
        sup_static   = np.column_stack([sup_static, sup_agg.reshape(-1, 1)])
        static_names.append(f"{var}_{WINDOW_DAYS}d_{agg_func}")

        ds.close()

    print("Saving...")
    np.savez_compressed(OUTPUT_FILE,
        t0_date=str(t0),
        static_feature_names=static_names,
        event_coords=event_coords,
        event_fold=event_fold,
        event_day_index=event_day_idx,
        event_static=event_static,
        event_bias=bias_event,
        support_coords=sup_coords,
        support_weights=sup_weights,
        support_static=sup_static,
        support_bias=bias_sup,
        unique_day_index=unique_days,
        **temporal_series
    )
    print(f"Wrote {OUTPUT_FILE}")
    print("Static features:", static_names)

# ----------------------------------------------------------------------
# Data loading (used by train.py)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SplitData:
    event_x: np.ndarray
    event_offset: np.ndarray
    support_x: np.ndarray
    support_offset: np.ndarray
    support_log_weights: np.ndarray
    unique_day_index: np.ndarray

# These will be populated when the module is imported
static_feature_names = []
event_static = None
event_bias = None
event_fold = None
event_day_index = None
support_static = None
support_bias = None
support_weights = None
unique_day_index = None

def _load_bundle():
    global static_feature_names, event_static, event_bias, event_fold, event_day_index
    global support_static, support_bias, support_weights, unique_day_index

    if not OUTPUT_FILE.exists():
        print(f"Data file {OUTPUT_FILE} not found. Generating now...")
        generate_data()

    d = np.load(OUTPUT_FILE, allow_pickle=True)
    static_feature_names = [str(x) for x in d["static_feature_names"]]
    event_static = d["event_static"].astype(np.float32)
    event_bias = d["event_bias"].astype(np.float32)
    event_fold = d["event_fold"].astype(np.int16)
    event_day_index = d["event_day_index"].astype(np.int32)
    support_static = d["support_static"].astype(np.float32)
    support_bias = d["support_bias"].astype(np.float32)
    support_weights = d["support_weights"].astype(np.float32)
    unique_day_index = d["unique_day_index"]

def build_point_process_splits(selected_static, selected_temporal_feats=None, val_fold=3):
    """Return train/val splits for given static feature names."""
    if event_static is None:
        _load_bundle()

    train_mask = event_fold != val_fold
    val_mask = event_fold == val_fold

    static_idx = [static_feature_names.index(n) for n in selected_static]
    e_static = event_static[:, static_idx]
    s_static = support_static[:, static_idx]

    e_x = e_static
    s_x = s_static.reshape(len(unique_day_index), -1, len(selected_static))

    # Standardize
    train_e = e_x[train_mask]
    train_s = s_x.reshape(-1, s_x.shape[-1])
    fit_x = np.vstack([train_e, train_s])
    mu = np.nanmean(fit_x, axis=0)
    sd = np.nanstd(fit_x, axis=0)
    sd[sd < 1e-6] = 1.0

    def norm(x):
        return np.nan_to_num((x - mu) / sd, nan=0.0)

    train_split = SplitData(
        event_x=norm(e_x[train_mask]),
        event_offset=np.log(np.clip(event_bias[train_mask], 1e-6, None)),
        support_x=norm(s_x),
        support_offset=np.log(np.clip(support_bias, 1e-6, None)),
        support_log_weights=np.log(np.clip(support_weights, 1e-6, None)),
        unique_day_index=unique_day_index,
    )
    val_split = SplitData(
        event_x=norm(e_x[val_mask]),
        event_offset=np.log(np.clip(event_bias[val_mask], 1e-6, None)),
        support_x=norm(s_x),
        support_offset=np.log(np.clip(support_bias, 1e-6, None)),
        support_log_weights=np.log(np.clip(support_weights, 1e-6, None)),
        unique_day_index=unique_day_index,
    )

    return train_split, val_split, selected_static, mu, sd

# ----------------------------------------------------------------------
# Main guard: run generation if executed as script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    generate_data()