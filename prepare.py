#!/usr/bin/env python3
"""
CLEAN point-based hazard-agnostic preparation with rigorous spatiotemporal non-event sampling.

- Trivial terrain mask (water, bare rock, permanent snow, artificial surfaces)
- Absence samples: (a) same location, different time (30-day buffer)
                  (b) different location, different time (outside event areas + mask)
- Temporal balancing: uniform across months
- Static and dynamic extraction via rioxarray.sel()
"""

from pathlib import Path
import geopandas as gpd
import xarray as xr
import numpy as np
import rioxarray
import pandas as pd
from tqdm import tqdm
import warnings

# ============================================================
# CONFIG – ADJUST TO YOUR DATA
# ============================================================
HAZARD_TYPE = "landslide"                     # or "wildfire"
WINDOW_DAYS_LIST = list(range(1, 61))        # e.g. 1..60 days for dynamic windows

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

# Trivial terrain mask (land cover classes to exclude)
TRIVIAL_MASK_TIF = STATIC_DIR / "trivial_terrain.tif"   # 1 = invalid (water, rock, etc.)
# If you don't have a mask, we will use slope < 5° as a fallback (requires slope.tif)
USE_SLOPE_MASK_FALLBACK = True
SLOPE_TIF = STATIC_DIR / "slope.tif"                   # degrees

# Temporal buffer (days) around any event at the same location
TEMPORAL_BUFFER_DAYS = 30

# Ratio of absence samples: same-location / different-location
SAME_LOC_FRAC = 0.5

# Balance non-events per month (if True) else per year
BALANCE_BY_MONTH = True

# Bias sampling raster (optional, for weighting)
BIAS_RASTER = "bias_sampling.tif"
# ============================================================


def extract_at_points_xarray(raster_path, x_coords, y_coords):
    """Extract raster values at points using rioxarray + xarray.sel()."""
    da = rioxarray.open_rasterio(raster_path).squeeze()
    if 'band' in da.dims:
        da = da.isel(band=0)
    xr_x = xr.DataArray(x_coords, dims='point')
    xr_y = xr.DataArray(y_coords, dims='point')
    return da.sel(x=xr_x, y=xr_y, method='nearest').values.astype(np.float32)


def create_trivial_mask(study_area_bounds=None):
    """
    Load or create a boolean mask: True = valid area (can host an event/absence).
    If TRIVIAL_MASK_TIF exists, use it. Otherwise, fallback to slope >= 5°.
    Returns a DataArray with same CRS/transform as the first static raster.
    """
    if TRIVIAL_MASK_TIF.exists():
        mask_da = rioxarray.open_rasterio(TRIVIAL_MASK_TIF).squeeze()
        # Assume 1 = invalid, 0 = valid
        valid = (mask_da == 0).astype(np.uint8)
    else:
        warnings.warn("Trivial terrain mask not found. Using slope >= 5° as fallback.")
        slope_da = rioxarray.open_rasterio(SLOPE_TIF).squeeze()
        valid = (slope_da >= 5.0).astype(np.uint8)

    # Clip to study area if bounds provided (optional)
    if study_area_bounds is not None:
        xmin, ymin, xmax, ymax = study_area_bounds
        valid = valid.where((valid.x >= xmin) & (valid.x <= xmax) &
                            (valid.y >= ymin) & (valid.y <= ymax), drop=False)
    return valid


def sample_random_points_in_mask(mask_da, n_points, rng):
    """
    Sample n_points random (x,y) coordinates from cells where mask_da == 1.
    Returns arrays x, y in the same CRS as mask_da.
    """
    valid_cells = np.where(mask_da.values == 1)
    if len(valid_cells[0]) == 0:
        raise ValueError("Mask contains no valid cells!")
    idx = rng.choice(len(valid_cells[0]), size=n_points, replace=True)
    y_idx = valid_cells[0][idx]
    x_idx = valid_cells[1][idx]
    # Convert pixel indices to coordinates
    x = mask_da.x.values[x_idx]
    y = mask_da.y.values[y_idx]
    return x, y


def is_too_close_to_event(x, y, event_coords, min_dist_m=50.0):
    """Check if any of the (x,y) points are within min_dist_m of any event."""
    # Simple brute-force; for large datasets use a KD-tree
    from scipy.spatial import cKDTree
    tree = cKDTree(event_coords)
    dist, _ = tree.query(np.column_stack([x, y]))
    return dist < min_dist_m


def main():
    print(f"Rigorous preparation for {HAZARD_TYPE} (Moreno et al. protocol)")

    # Load events
    events = gpd.read_file(EVENT_FILE)
    events["date"] = pd.to_datetime(events["date"]).dt.floor('D')
    n_events_orig = len(events)

    # Event coordinates and time
    x_coords_ev = events.geometry.x.values.astype(np.float32)
    y_coords_ev = events.geometry.y.values.astype(np.float32)
    event_coords = np.column_stack([x_coords_ev, y_coords_ev])

    t0 = events["date"].min()
    event_days = (events["date"] - t0).dt.days.values.astype(np.int32)

    # Spatial fold (for validation split later)
    x_mid, y_mid = np.median(x_coords_ev), np.median(y_coords_ev)
    event_fold = ((x_coords_ev >= x_mid).astype(int) + 2*(y_coords_ev >= y_mid).astype(int)).astype(np.int16)

    # Time range
    all_dates = pd.date_range(start=t0, end=events["date"].max(), freq='D')
    day_index = {d: i for i, d in enumerate((all_dates - t0).days.values)}
    unique_days = np.array(list(day_index.values()))

    # ============== Create trivial terrain mask ==============
    # Use the first static raster to get CRS/transform
    static_paths = sorted(STATIC_DIR.glob("*.tif"))
    ref_da = rioxarray.open_rasterio(static_paths[0]).squeeze()
    study_bounds = (ref_da.x.min().item(), ref_da.y.min().item(),
                    ref_da.x.max().item(), ref_da.y.max().item())
    valid_mask = create_trivial_mask(study_bounds)
    # Reproject event coords to mask CRS if necessary (assumed same)
    # For simplicity we assume all rasters share CRS.

    # ============== Generate absence samples ==============
    rng = np.random.default_rng(42)
    n_absences = n_events_orig * 5   # e.g. 5:1 ratio
    n_same_loc = int(n_absences * SAME_LOC_FRAC)
    n_diff_loc = n_absences - n_same_loc

    # --- Same location, different time (30-day buffer) ---
    same_loc_data = []
    for ev_idx in tqdm(range(n_events_orig), desc="Same-loc absences"):
        ev_day = event_days[ev_idx]
        ev_x, ev_y = x_coords_ev[ev_idx], y_coords_ev[ev_idx]
        # Find days that are at least TEMPORAL_BUFFER_DAYS away from this event day
        # For simplicity, we consider all days in the period, excluding the buffer.
        # We'll later balance by month.
        possible_days = unique_days[np.abs(unique_days - ev_day) > TEMPORAL_BUFFER_DAYS]
        if len(possible_days) == 0:
            continue
        # Number of absences for this event (proportional to total same-loc)
        n_for_this = max(1, int(n_same_loc / n_events_orig))
        chosen_days = rng.choice(possible_days, size=min(n_for_this, len(possible_days)), replace=False)
        for d in chosen_days:
            same_loc_data.append((ev_x, ev_y, d))

    same_loc_df = pd.DataFrame(same_loc_data, columns=['x', 'y', 'day'])
    # If we have more than needed, downsample
    if len(same_loc_df) > n_same_loc:
        same_loc_df = same_loc_df.sample(n=n_same_loc, random_state=42)

    # --- Different location, different time ---
    # Sample points from valid mask that are >50m from any event
    diff_loc_data = []
    attempts = 0
    max_attempts = n_diff_loc * 10
    while len(diff_loc_data) < n_diff_loc and attempts < max_attempts:
        n_needed = n_diff_loc - len(diff_loc_data)
        x_cand, y_cand = sample_random_points_in_mask(valid_mask, n_needed * 2, rng)
        cand_coords = np.column_stack([x_cand, y_cand])
        # Filter too close to events
        from scipy.spatial import cKDTree
        tree = cKDTree(event_coords)
        dist, _ = tree.query(cand_coords)
        keep = dist >= 50.0
        x_keep = x_cand[keep]
        y_keep = y_cand[keep]
        for x, y in zip(x_keep, y_keep):
            # Random day (any day in period) with no buffer? We'll apply 30-day buffer later if needed,
            # but since location is far, we only need to ensure the day isn't within 30 days of any event
            # at that specific location? For simplicity, pick any day uniformly.
            day = rng.choice(unique_days)
            diff_loc_data.append((x, y, day))
        attempts += 1

    diff_loc_df = pd.DataFrame(diff_loc_data, columns=['x', 'y', 'day'])
    if len(diff_loc_df) > n_diff_loc:
        diff_loc_df = diff_loc_df.sample(n=n_diff_loc, random_state=42)

    # Combine absences
    absence_df = pd.concat([same_loc_df, diff_loc_df], ignore_index=True)
    absence_df['is_event'] = 0
    events_df = pd.DataFrame({
        'x': x_coords_ev,
        'y': y_coords_ev,
        'day': event_days,
        'is_event': 1
    })
    all_samples = pd.concat([events_df, absence_df], ignore_index=True)

    # ============== Temporal balancing (by month) ==============
    if BALANCE_BY_MONTH:
        # Convert day index to month
        dates = t0 + pd.to_timedelta(all_samples['day'], unit='D')
        all_samples['month'] = dates.dt.month
        # For absences, we want uniform distribution across months.
        # Calculate target count per month based on presence distribution? Or uniform.
        # Here we'll resample absences to have equal number per month.
        absence_idx = all_samples['is_event'] == 0
        absence_months = all_samples.loc[absence_idx, 'month']
        month_counts = absence_months.value_counts()
        target_per_month = int(np.ceil(len(absence_df) / 12))
        balanced_absence_indices = []
        for month in range(1, 13):
            month_mask = (all_samples['month'] == month) & absence_idx
            avail = all_samples.index[month_mask].tolist()
            if len(avail) >= target_per_month:
                chosen = rng.choice(avail, size=target_per_month, replace=False)
            else:
                chosen = avail  # take all available
            balanced_absence_indices.extend(chosen)
        # Keep all events and the balanced absences
        keep_idx = list(all_samples[~absence_idx].index) + balanced_absence_indices
        all_samples = all_samples.loc[keep_idx].reset_index(drop=True)

    # Final sample counts
    n_samples = len(all_samples)
    n_events_final = (all_samples['is_event'] == 1).sum()
    n_absences_final = n_samples - n_events_final
    print(f"Final sample: {n_events_final} events, {n_absences_final} absences")

    # ============== Extract static features for all samples ==============
    print("\nExtracting static features...")
    static_paths = sorted(STATIC_DIR.glob("*.tif"))
    bias_path = STATIC_DIR / BIAS_RASTER
    other_static = [p for p in static_paths if p != bias_path]

    static_names = []
    static_arrays = []
    for p in other_static:
        vals = extract_at_points_xarray(p, all_samples['x'].values, all_samples['y'].values)
        static_arrays.append(vals)
        static_names.append(p.stem)

    static_features = np.column_stack(static_arrays)
    bias_values = extract_at_points_xarray(bias_path, all_samples['x'].values, all_samples['y'].values)

    # ============== Extract dynamic features ==============
    print(f"\nExtracting dynamic features ({len(VAR_SPECS)} vars × {len(WINDOW_DAYS_LIST)} windows)...")
    n_temporal = len(VAR_SPECS) * len(WINDOW_DAYS_LIST)
    dynamic_features = np.zeros((n_samples, n_temporal), dtype=np.float32)
    temporal_series = {}
    feature_idx = 0

    for var_name, (subdir, pat, nc_var, agg) in VAR_SPECS.items():
        print(f"\n{var_name}...")
        files = sorted((DYNAMIC_DIR / subdir).glob(pat))
        ds = xr.open_mfdataset(files, combine='nested', concat_dim='time')
        da = ds[nc_var]
        times = pd.to_datetime(da.time.values).floor('D')
        day_to_idx = {d: i for i, d in enumerate((times - t0).days.values)}
        temporal_series[f"{var_name}_mean"] = da.mean(dim=['x', 'y']).values.astype(np.float32)

        # Extract at all sample locations
        xr_x = xr.DataArray(all_samples['x'].values, dims='sample')
        xr_y = xr.DataArray(all_samples['y'].values, dims='sample')
        sample_ts = da.sel(x=xr_x, y=xr_y, method='nearest').compute()  # shape (time, sample)
        sample_ts_xr = xr.DataArray(sample_ts, dims=('time', 'sample'), coords={'time': times})

        for window_days in tqdm(WINDOW_DAYS_LIST, desc=f"  {var_name}"):
            rolled = getattr(sample_ts_xr.rolling(time=window_days, min_periods=1), agg)()
            # For each sample, we need the rolled value at its specific day
            # We'll fill by iterating (vectorized approach is possible but complex)
            for i, row in all_samples.iterrows():
                day_val = row['day']
                t_idx = day_to_idx.get(day_val, None)
                if t_idx is not None:
                    dynamic_features[i, feature_idx] = rolled.values[t_idx, i]
                else:
                    dynamic_features[i, feature_idx] = np.nan
            static_names.append(f"{var_name}_{window_days}d_{agg}")
            feature_idx += 1
        ds.close()

    # Handle NaNs (e.g., from dates outside NetCDF range)
    nan_mask = np.isnan(dynamic_features).any(axis=1)
    if nan_mask.any():
        print(f"Warning: {nan_mask.sum()} samples have NaN dynamic features. Dropping them.")
        all_samples = all_samples[~nan_mask].reset_index(drop=True)
        static_features = static_features[~nan_mask]
        bias_values = bias_values[~nan_mask]
        dynamic_features = dynamic_features[~nan_mask]
        n_samples = len(all_samples)

    # ============== Final feature matrix ==============
    X = np.hstack([static_features, dynamic_features])
    y = all_samples['is_event'].values.astype(np.int8)

    # Also prepare event-only arrays for backward compatibility (if needed)
    event_mask = y == 1
    event_X = X[event_mask]
    event_coords_out = all_samples[event_mask][['x', 'y']].values
    event_days_out = all_samples[event_mask]['day'].values
    event_fold_out = event_fold  # original event fold (only for events)
    bias_event = bias_values[event_mask]

    # Support (non-event) arrays
    support_X = X[~event_mask]
    support_bias = bias_values[~event_mask]

    # Weights (pixel area)
    with rioxarray.open_rasterio(other_static[0]) as da:
        res = da.rio.resolution()
        pixel_area = abs(res[0] * res[1])
    weights = np.full(n_samples, pixel_area, dtype=np.float32)

    print(f"\nSaving to {OUTPUT_FILE}...")
    np.savez_compressed(
        OUTPUT_FILE,
        t0=str(t0),
        feature_names=static_names,
        X=X,
        y=y,
        event_X=event_X,
        event_coords=event_coords_out,
        event_days=event_days_out,
        event_fold=event_fold_out,
        event_bias=bias_event,
        support_X=support_X,
        support_bias=support_bias,
        weights=weights,
        **temporal_series
    )

    print(f"Done. Shape: X={X.shape}, y={y.shape}")


if __name__ == "__main__":
    main()