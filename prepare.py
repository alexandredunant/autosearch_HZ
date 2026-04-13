#!/usr/bin/env python3
"""
CLEAN point-based hazard-agnostic preparation.
Uses rioxarray + xarray.sel() for ALL raster extraction (static + dynamic).
"""

from pathlib import Path
import geopandas as gpd
import xarray as xr
import numpy as np
import rioxarray  # Adds .rio accessor to xarray
import pandas as pd
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
HAZARD_TYPE = "landslide"
WINDOW_DAYS_LIST = list(range(1, 61))

VAR_SPECS = {
    "precipitation": ("Precipitation", "prec_*.nc", "prec", "sum"),
    "temperature":   ("Temperature", "tmean_*.nc", "tmean", "mean"),
    "lightning":     ("Lightning_Standardized", "lightning_density_*.nc", "lightning_density", "max"),
}

ROOT = Path(__file__).resolve().parent / "autosearch_data"
EVENT_FILE = ROOT / f"hazards_{HAZARD_TYPE}s.gpkg"
STATIC_DIR = ROOT / "static"
DYNAMIC_DIR = ROOT / "dynamic"
OUTPUT_FILE = ROOT / f"prepared_clean_{HAZARD_TYPE}.npz"
BIAS_RASTER = "bias_sampling.tif"
# ============================================================


def extract_at_points_xarray(raster_path, x_coords, y_coords):
    """
    Extract raster values at points using rioxarray + xarray.sel().
    Works for both GeoTIFF (via rioxarray) and NetCDF.
    """
    # Open with rioxarray (adds .rio accessor for CRS/transform handling)
    da = rioxarray.open_rasterio(raster_path).squeeze()

    # Handle both (band, y, x) and (y, x) shapes
    if 'band' in da.dims:
        da = da.isel(band=0)

    # xarray point selection (fast, vectorized)
    xr_x = xr.DataArray(x_coords, dims='point')
    xr_y = xr.DataArray(y_coords, dims='point')

    return da.sel(x=xr_x, y=xr_y, method='nearest').values.astype(np.float32)


def main():
    print(f"Clean point-based preparation for {HAZARD_TYPE}")

    # Load events
    events = gpd.read_file(EVENT_FILE)
    events["date"] = pd.to_datetime(events["date"]).dt.floor('D')

    # Event coordinates
    x_coords = events.geometry.x.values.astype(np.float32)
    y_coords = events.geometry.y.values.astype(np.float32)
    coords = np.column_stack([x_coords, y_coords])
    n_events = len(events)

    # Spatial fold
    x_mid, y_mid = np.median(x_coords), np.median(y_coords)
    event_fold = ((x_coords >= x_mid).astype(int) + 2*(y_coords >= y_mid).astype(int)).astype(np.int16)

    # Time
    t0 = events["date"].min()
    event_days = (events["date"] - t0).dt.days.values.astype(np.int32)
    unique_days = np.unique(event_days)
    n_days = len(unique_days)

    print(f"Events: {n_events}, Days: {n_days}")

    # ============== STATIC (using xarray.sel like dynamic) ==============
    print("\nStatic features (using rioxarray + xarray.sel)...")
    static_paths = sorted(STATIC_DIR.glob("*.tif"))
    bias_path = STATIC_DIR / BIAS_RASTER
    other_static = [p for p in static_paths if p != bias_path]

    # Extract all static rasters using xarray.sel (consistent with dynamic!)
    static_list = []
    static_names = []

    for p in other_static:
        vals = extract_at_points_xarray(p, x_coords, y_coords)
        static_list.append(vals)
        static_names.append(p.stem)

    event_static = np.column_stack(static_list)
    bias_event = extract_at_points_xarray(bias_path, x_coords, y_coords)

    print(f"  {len(static_names)} static features extracted")

    # ============== DYNAMIC (same method, just with rolling) ==============
    print(f"\nDynamic features ({len(VAR_SPECS)} vars × {len(WINDOW_DAYS_LIST)} windows)...")

    n_static = len(static_names)
    n_temporal = len(VAR_SPECS) * len(WINDOW_DAYS_LIST)

    event_dynamic = np.zeros((n_events, n_temporal), dtype=np.float32)
    support_dynamic = np.zeros((n_days * n_events, n_temporal), dtype=np.float32)

    temporal_series = {}
    feature_idx = 0

    for var_name, (subdir, pat, nc_var, agg) in VAR_SPECS.items():
        print(f"\n{var_name}...")

        # Load NetCDF
        files = sorted((DYNAMIC_DIR / subdir).glob(pat))
        ds = xr.open_mfdataset(files, combine='nested', concat_dim='time')
        da = ds[nc_var]

        times = pd.to_datetime(da.time.values).floor('D')
        day_to_idx = {d: i for i, d in enumerate((times - t0).days.values)}
        temporal_series[f"{var_name}_mean"] = da.mean(dim=['x', 'y']).values.astype(np.float32)

        # Extract at event locations using xarray.sel (same as static!)
        xr_x = xr.DataArray(x_coords, dims='event')
        xr_y = xr.DataArray(y_coords, dims='event')
        event_ts = da.sel(x=xr_x, y=xr_y, method='nearest').compute()

        # Rolling windows
        event_ts_xr = xr.DataArray(event_ts, dims=('time', 'event'), coords={'time': times})

        for window_days in tqdm(WINDOW_DAYS_LIST, desc=f"  {var_name}"):
            rolled = getattr(event_ts_xr.rolling(time=window_days, min_periods=1), agg)()

            # Event: value at event day
            t_idx = np.array([day_to_idx.get(d, -1) for d in event_days])
            valid = t_idx >= 0
            if valid.any():
                event_dynamic[valid, feature_idx] = rolled.values[t_idx[valid], np.arange(n_events)[valid]]

            # Support: all days × all events
            day_idx = np.array([day_to_idx.get(d, -1) for d in unique_days])
            valid_days = day_idx >= 0
            if valid_days.any():
                rolled_days = rolled.values[day_idx[valid_days]]
                for i, day_rolled in enumerate(rolled_days):
                    start = i * n_events
                    end = start + n_events
                    support_dynamic[start:end, feature_idx] = day_rolled

            static_names.append(f"{var_name}_{window_days}d_{agg}")
            feature_idx += 1

        ds.close()

    # Combine
    event_full = np.hstack([event_static, event_dynamic])
    support_static = np.tile(event_static, (n_days, 1))
    support_full = np.hstack([support_static, support_dynamic])

    # Weights from first static raster resolution
    with rioxarray.open_rasterio(other_static[0]) as da:
        res = da.rio.resolution()
        pixel_area = abs(res[0] * res[1])
    support_weights = np.full(n_days * n_events, pixel_area, dtype=np.float32)

    # Save
    print(f"\nSaving...")
    np.savez_compressed(
        OUTPUT_FILE,
        t0=str(t0),
        feature_names=static_names,
        n_events=n_events,
        n_days=n_days,
        event_coords=coords,
        event_days=event_days,
        event_fold=event_fold,
        event_static=event_full,
        event_bias=bias_event,
        support_static=support_full,
        support_bias=np.tile(bias_event, n_days),
        support_weights=support_weights,
        unique_days=unique_days,
        **temporal_series
    )

    print(f"Saved {OUTPUT_FILE}")
    print(f"  Event: {event_full.shape}, Support: {support_full.shape}")
    print(f"  Features: {len(static_names)}")


if __name__ == "__main__":
    main()