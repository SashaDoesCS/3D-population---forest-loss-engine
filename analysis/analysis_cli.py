#!/usr/bin/env python3
"""
FOREST LOSS SPATIAL ANALYSIS - CLI Tool
========================================

Command-line interface for analyzing forest loss correlations.
Used by Electron app for interactive analysis.

Features:
- Point-based analysis (click on globe)
- Region-based analysis (predefined regions)
- Distance-decay correlation analysis
- Directional (anisotropy) analysis
- Global hotspot scanning

Output: JSON results file for visualization
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
from scipy import stats

# Suppress warnings during correlation calculations
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='An input array is constant')

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CONFIGURATION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

DISTANCE_BANDS = [0, 2000, 5000, 10000, 20000, 50000, 100000]  # meters
DIRECTIONAL_BANDS = [0, 10000, 25000, 50000, 100000]  # meters for directional analysis
DIRECTIONS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
MIN_BINS_FOR_CORRELATION = 20  # Minimum data points for meaningful correlation
MIN_POPULATION = 0.1  # Minimum population to consider


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# JSON SERIALIZATION - CRITICAL FIX FOR numpy.bool_ ERROR
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def to_json_serializable(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    
    This fixes the 'Object of type bool is not JSON serializable' error.
    """
    if obj is None:
        return None
    
    # Handle numpy boolean FIRST (most specific)
    # Note: np.bool_ is the correct type, np.bool8 was removed in numpy 2.0
    if isinstance(obj, np.bool_):
        return bool(obj)
    
    # Handle numpy integers
    if isinstance(obj, np.integer):
        return int(obj)
    
    # Handle numpy floats
    if isinstance(obj, np.floating):
        val = float(obj)
        # Handle NaN and Inf
        if np.isnan(val):
            return None
        if np.isinf(val):
            return None
        return val
    
    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return [to_json_serializable(x) for x in obj.tolist()]
    
    # Handle pandas Series
    if isinstance(obj, pd.Series):
        return to_json_serializable(obj.to_dict())
    
    # Handle pandas DataFrame
    if isinstance(obj, pd.DataFrame):
        return to_json_serializable(obj.to_dict())
    
    # Handle dictionaries
    if isinstance(obj, dict):
        return {str(k): to_json_serializable(v) for k, v in obj.items()}
    
    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(x) for x in obj]
    
    # Handle Python native types
    if isinstance(obj, (bool, int, float, str)):
        return obj
    
    # Fallback: convert to string
    try:
        return str(obj)
    except:
        return None


def save_json(data, filepath):
    """Save data to JSON with proper serialization."""
    serializable = to_json_serializable(data)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# DISTANCE CALCULATIONS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between points in meters.
    Vectorized for efficiency with numpy arrays.
    """
    R = 6371000  # Earth's radius in meters
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.clip(np.sqrt(a), 0, 1))
    
    return R * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate bearing from point 1 to point 2 in degrees (0-360).
    0 = North, 90 = East, 180 = South, 270 = West
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


def bearing_to_direction(bearing):
    """Convert bearing (0-360) to cardinal direction."""
    # N: 337.5-22.5, NE: 22.5-67.5, E: 67.5-112.5, etc.
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    idx = int((bearing + 22.5) / 45) % 8
    return directions[idx]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# DATA LOADING
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def parse_tile_name(tile_name):
    """
    Parse tile name like '50N_130W' into lat/lon bounds.
    Returns (min_lat, max_lat, min_lon, max_lon)
    """
    import re
    match = re.match(r'(\d+)([NS])_(\d+)([EW])', tile_name)
    if not match:
        return None
    
    lat = int(match.group(1))
    if match.group(2) == 'S':
        lat = -lat
    
    lon = int(match.group(3))
    if match.group(4) == 'W':
        lon = -lon
    
    # Hansen tiles are 10x10 degrees, with the tile name indicating the upper-left corner
    return (lat - 10, lat, lon, lon + 10)


def find_tiles_for_bounds(tiles_dir, min_lat, max_lat, min_lon, max_lon):
    """Find all tiles that intersect with the given bounds."""
    tiles = []
    
    if not os.path.exists(tiles_dir):
        print(f"[Warning] Tiles directory not found: {tiles_dir}")
        return tiles
    
    for tile_name in os.listdir(tiles_dir):
        tile_path = os.path.join(tiles_dir, tile_name)
        if not os.path.isdir(tile_path):
            continue
        
        bounds = parse_tile_name(tile_name)
        if bounds is None:
            continue
        
        tile_min_lat, tile_max_lat, tile_min_lon, tile_max_lon = bounds
        
        # Check for intersection
        if (tile_min_lat <= max_lat and tile_max_lat >= min_lat and
            tile_min_lon <= max_lon and tile_max_lon >= min_lon):
            tiles.append(tile_name)
    
    return tiles


def load_tile_data(tiles_dir, tile_name, year):
    """Load CSV data for a specific tile and year."""
    csv_path = os.path.join(tiles_dir, tile_name, f"canopy_population_loss_{year}.csv")
    
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        # Ensure required columns exist
        required = ['lat', 'lon', 'canopy_mean', 'lossyear_mean']
        if not all(col in df.columns for col in required):
            print(f"[Warning] Missing columns in {csv_path}")
            return None
        
        # Handle different population column names
        if 'population_density' in df.columns:
            df['population_mean'] = df['population_density']
        elif 'population_count' in df.columns:
            df['population_mean'] = df['population_count']
        elif 'population_mean' not in df.columns:
            df['population_mean'] = 0
        
        return df
        
    except Exception as e:
        print(f"[Warning] Failed to load {csv_path}: {e}")
        return None


def load_data_for_point(tiles_dir, center_lat, center_lon, radius_km, year):
    """
    Load data from all tiles within radius of center point.
    """
    # Calculate bounding box (rough approximation)
    lat_delta = radius_km / 111  # ~111 km per degree latitude
    lon_delta = radius_km / (111 * np.cos(np.radians(center_lat)))
    
    min_lat = center_lat - lat_delta
    max_lat = center_lat + lat_delta
    min_lon = center_lon - lon_delta
    max_lon = center_lon + lon_delta
    
    # Find relevant tiles
    tiles = find_tiles_for_bounds(tiles_dir, min_lat, max_lat, min_lon, max_lon)
    
    if not tiles:
        print(f"[Warning] No tiles found for bounds: {min_lat:.2f}-{max_lat:.2f}, {min_lon:.2f}-{max_lon:.2f}")
        return None
    
    print(f"[Info] Loading data from {len(tiles)} tiles: {', '.join(tiles)}")
    
    # Load and combine data
    dfs = []
    for tile_name in tiles:
        df = load_tile_data(tiles_dir, tile_name, year)
        if df is not None and len(df) > 0:
            dfs.append(df)
    
    if not dfs:
        return None
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Filter to points within radius
    distances = haversine_distance(
        center_lat, center_lon,
        combined['lat'].values, combined['lon'].values
    )
    
    mask = distances <= (radius_km * 1000)
    filtered = combined[mask].copy()
    filtered['distance_m'] = distances[mask]
    
    # Calculate bearing for directional analysis
    bearings = calculate_bearing(
        center_lat, center_lon,
        filtered['lat'].values, filtered['lon'].values
    )
    filtered['bearing'] = bearings
    filtered['direction'] = [bearing_to_direction(b) for b in bearings]
    
    print(f"[Info] Loaded {len(filtered):,} points within {radius_km}km radius")
    
    return filtered


def load_data_for_region(tiles_dir, bounds, year):
    """Load data for a predefined region."""
    min_lat = bounds['minLat']
    max_lat = bounds['maxLat']
    min_lon = bounds['minLon']
    max_lon = bounds['maxLon']
    
    tiles = find_tiles_for_bounds(tiles_dir, min_lat, max_lat, min_lon, max_lon)
    
    if not tiles:
        print(f"[Warning] No tiles found for region")
        return None
    
    print(f"[Info] Loading data from {len(tiles)} tiles")
    
    dfs = []
    for tile_name in tiles:
        df = load_tile_data(tiles_dir, tile_name, year)
        if df is not None and len(df) > 0:
            # Filter to region bounds
            mask = (
                (df['lat'] >= min_lat) & (df['lat'] <= max_lat) &
                (df['lon'] >= min_lon) & (df['lon'] <= max_lon)
            )
            filtered = df[mask]
            if len(filtered) > 0:
                dfs.append(filtered)
    
    if not dfs:
        return None
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Calculate distance from region center for analysis
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    
    distances = haversine_distance(
        center_lat, center_lon,
        combined['lat'].values, combined['lon'].values
    )
    combined['distance_m'] = distances
    
    bearings = calculate_bearing(
        center_lat, center_lon,
        combined['lat'].values, combined['lon'].values
    )
    combined['bearing'] = bearings
    combined['direction'] = [bearing_to_direction(b) for b in bearings]
    
    print(f"[Info] Loaded {len(combined):,} points for region")
    
    return combined


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CORRELATION ANALYSIS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def safe_correlation(x, y):
    """
    Calculate Pearson correlation with error handling.
    Returns (correlation, p_value, is_valid)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    
    # Need enough data points
    if len(x) < MIN_BINS_FOR_CORRELATION:
        return None, None, False
    
    # Check for constant arrays (std = 0)
    if np.std(x) == 0 or np.std(y) == 0:
        return None, None, False
    
    try:
        corr, pval = stats.pearsonr(x, y)
        
        if np.isnan(corr) or np.isnan(pval):
            return None, None, False
        
        return float(corr), float(pval), True
        
    except Exception:
        return None, None, False


def analyze_distance_decay(df, distance_bands=DISTANCE_BANDS):
    """
    Analyze how correlation between population and forest loss decays with distance.
    """
    results = {
        'band_correlations': {},
        'half_distance': None
    }
    
    if df is None or len(df) == 0:
        return results
    
    # Prepare loss indicator (binary: had loss or not)
    loss_values = (df['lossyear_mean'] > 0).astype(float).values
    pop_values = df['population_mean'].values
    distances = df['distance_m'].values
    
    correlations = []
    bands_with_data = []
    
    for i, band_max in enumerate(distance_bands[1:]):
        band_min = distance_bands[i]
        
        # Select points in this distance band
        mask = (distances >= band_min) & (distances < band_max)
        
        band_loss = loss_values[mask]
        band_pop = pop_values[mask]
        
        if len(band_loss) >= MIN_BINS_FOR_CORRELATION:
            corr, pval, valid = safe_correlation(band_pop, band_loss)
            
            if valid:
                results['band_correlations'][int(band_max)] = {
                    'mean': corr,
                    'std': 0,  # Single calculation, no std
                    'count': int(len(band_loss)),
                    'p_value': pval
                }
                correlations.append(corr)
                bands_with_data.append(band_max)
    
    # Calculate half-distance (where correlation drops to 50% of max)
    if correlations and len(correlations) >= 2:
        max_corr = max(correlations)
        half_corr = max_corr * 0.5
        
        for i, (band, corr) in enumerate(zip(bands_with_data, correlations)):
            if corr <= half_corr:
                if i == 0:
                    half_dist = band / 2
                else:
                    # Linear interpolation
                    prev_band = bands_with_data[i-1]
                    prev_corr = correlations[i-1]
                    if prev_corr != corr:  # Avoid division by zero
                        half_dist = prev_band + (half_corr - prev_corr) * (band - prev_band) / (corr - prev_corr)
                    else:
                        half_dist = (prev_band + band) / 2
                
                results['half_distance'] = {
                    'distance_m': float(half_dist),
                    'distance_km': float(half_dist / 1000),
                    'max_correlation': float(max_corr),
                    'half_correlation': float(half_corr)
                }
                break
        
        # If correlation never drops below half
        if results['half_distance'] is None and correlations:
            results['half_distance'] = {
                'distance_m': float(bands_with_data[-1]),
                'distance_km': float(bands_with_data[-1] / 1000),
                'max_correlation': float(max_corr),
                'half_correlation': float(half_corr),
                'note': 'Correlation did not drop to 50%'
            }
    
    return results


def analyze_directional(df, distance_bands=DIRECTIONAL_BANDS):
    """
    Analyze directional patterns to test for isotropy.
    """
    results = {
        'directional': {},
        'isotropy_test': None
    }
    
    if df is None or len(df) == 0 or 'direction' not in df.columns:
        return results
    
    loss_values = (df['lossyear_mean'] > 0).astype(float).values
    pop_values = df['population_mean'].values
    distances = df['distance_m'].values
    directions = df['direction'].values
    
    # Analyze each direction
    direction_correlations = {d: [] for d in DIRECTIONS}
    
    for direction in DIRECTIONS:
        dir_mask = directions == direction
        
        results['directional'][direction] = {}
        
        for i, band_max in enumerate(distance_bands[1:]):
            band_min = distance_bands[i]
            
            band_mask = dir_mask & (distances >= band_min) & (distances < band_max)
            
            band_loss = loss_values[band_mask]
            band_pop = pop_values[band_mask]
            
            if len(band_loss) >= MIN_BINS_FOR_CORRELATION:
                corr, pval, valid = safe_correlation(band_pop, band_loss)
                
                if valid:
                    results['directional'][direction][int(band_max)] = {
                        'mean': corr,
                        'count': int(len(band_loss)),
                        'p_value': pval
                    }
                    direction_correlations[direction].append(corr)
    
    # Isotropy test using Levene's test for equal variances
    # and Kruskal-Wallis for differences in medians
    all_correlations = [corrs for corrs in direction_correlations.values() if len(corrs) > 0]
    
    if len(all_correlations) >= 4:  # Need at least 4 directions with data
        try:
            # Flatten for Levene's test
            flat_corrs = [c for corrs in all_correlations for c in corrs]
            
            # Levene's test (checks if variances are equal)
            if len(all_correlations) >= 2 and all(len(c) >= 2 for c in all_correlations[:2]):
                levene_stat, levene_p = stats.levene(*[c for c in all_correlations if len(c) >= 2])
                
                # Kruskal-Wallis test (non-parametric check for differences)
                kw_stat, kw_p = stats.kruskal(*[c for c in all_correlations if len(c) >= 2])
                
                # Is isotropic if p > 0.05 (no significant difference)
                is_isotropic = bool(levene_p > 0.05 and kw_p > 0.05)
                
                results['isotropy_test'] = {
                    'is_isotropic': is_isotropic,
                    'levene_p': float(levene_p),
                    'levene_stat': float(levene_stat),
                    'kruskal_p': float(kw_p),
                    'kruskal_stat': float(kw_stat),
                    'directions_analyzed': len(all_correlations)
                }
        except Exception as e:
            print(f"[Warning] Isotropy test failed: {e}")
            results['isotropy_test'] = {
                'is_isotropic': True,  # Default assumption
                'error': str(e)
            }
    else:
        results['isotropy_test'] = {
            'is_isotropic': True,  # Default when insufficient data
            'note': 'Insufficient directional data for test'
        }
    
    return results


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MAIN ANALYSIS FUNCTIONS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def run_point_analysis(tiles_dir, lat, lon, radius_km, year, output_dir):
    """Run analysis for a clicked point."""
    print(f"\n{'='*60}")
    print(f"POINT ANALYSIS")
    print(f"Center: ({lat:.4f}, {lon:.4f})")
    print(f"Radius: {radius_km} km")
    print(f"Year: {year}")
    print(f"{'='*60}\n")
    
    # Load data
    df = load_data_for_point(tiles_dir, lat, lon, radius_km, year)
    
    if df is None or len(df) == 0:
        results = {
            'error': 'No data found for this location',
            'metadata': {
                'center_lat': lat,
                'center_lon': lon,
                'radius_km': radius_km,
                'year': year,
                'n_data_points': 0
            }
        }
        save_json(results, os.path.join(output_dir, 'results.json'))
        return results
    
    # Run analyses
    print("[Analysis] Distance-decay analysis...")
    distance_decay = analyze_distance_decay(df)
    
    print("[Analysis] Directional analysis...")
    directional = analyze_directional(df)
    
    # Compile results
    half_dist_km = None
    if distance_decay.get('half_distance'):
        half_dist_km = distance_decay['half_distance'].get('distance_km')
    
    results = {
        'half_distance_km': half_dist_km,
        'distance_decay': distance_decay,
        'directional': directional,
        'summary': {
            'total_bins': len(df),
            'bins_with_loss': int((df['lossyear_mean'] > 0).sum()),
            'loss_percentage': float((df['lossyear_mean'] > 0).mean() * 100),
            'mean_canopy': float(df['canopy_mean'].mean()),
            'mean_population': float(df['population_mean'].mean()),
            'total_population': float(df['population_mean'].sum())
        },
        'metadata': {
            'center_lat': lat,
            'center_lon': lon,
            'radius_km': radius_km,
            'year': year,
            'n_data_points': len(df),
            'timestamp': datetime.now().isoformat()
        }
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    save_json(results, os.path.join(output_dir, 'results.json'))
    
    print(f"\n[Done] Results saved to {output_dir}/results.json")
    
    return results


def run_region_analysis(tiles_dir, region_id, regions_file, year, output_dir):
    """Run analysis for a predefined region."""
    # Load region definition
    with open(regions_file, 'r') as f:
        regions = json.load(f)
    
    if region_id not in regions:
        raise ValueError(f"Unknown region: {region_id}")
    
    region = regions[region_id]
    bounds = region['bounds']
    
    print(f"\n{'='*60}")
    print(f"REGION ANALYSIS: {region['name']}")
    print(f"Bounds: {bounds}")
    print(f"Year: {year}")
    print(f"{'='*60}\n")
    
    # Load data
    df = load_data_for_region(tiles_dir, bounds, year)
    
    if df is None or len(df) == 0:
        results = {
            'error': 'No data found for this region',
            'metadata': {
                'region_id': region_id,
                'region_name': region['name'],
                'year': year,
                'n_data_points': 0
            }
        }
        save_json(results, os.path.join(output_dir, 'results.json'))
        return results
    
    # Run analyses
    print("[Analysis] Distance-decay analysis...")
    distance_decay = analyze_distance_decay(df)
    
    print("[Analysis] Directional analysis...")
    directional = analyze_directional(df)
    
    # Compile results
    half_dist_km = None
    if distance_decay.get('half_distance'):
        half_dist_km = distance_decay['half_distance'].get('distance_km')
    
    results = {
        'half_distance_km': half_dist_km,
        'distance_decay': distance_decay,
        'directional': directional,
        'summary': {
            'total_bins': len(df),
            'bins_with_loss': int((df['lossyear_mean'] > 0).sum()),
            'loss_percentage': float((df['lossyear_mean'] > 0).mean() * 100),
            'mean_canopy': float(df['canopy_mean'].mean()),
            'mean_population': float(df['population_mean'].mean()),
            'total_population': float(df['population_mean'].sum())
        },
        'metadata': {
            'region_id': region_id,
            'region_name': region['name'],
            'bounds': bounds,
            'year': year,
            'n_data_points': len(df),
            'timestamp': datetime.now().isoformat()
        }
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    save_json(results, os.path.join(output_dir, 'results.json'))
    
    print(f"\n[Done] Results saved to {output_dir}/results.json")
    
    return results


def run_global_hotspot_scan(tiles_dir, year, output_dir, grid_size_deg=5.0):
    """
    Scan the globe for hotspots of population-forest loss correlation.
    This is useful for research to identify areas of interest.
    
    Returns regions ranked by correlation strength.
    """
    print(f"\n{'='*60}")
    print(f"GLOBAL HOTSPOT SCAN")
    print(f"Year: {year}")
    print(f"Grid size: {grid_size_deg}Â°")
    print(f"{'='*60}\n")
    
    hotspots = []
    
    # Scan in grid pattern
    for lat in np.arange(-60, 70, grid_size_deg):
        for lon in np.arange(-180, 180, grid_size_deg):
            center_lat = lat + grid_size_deg / 2
            center_lon = lon + grid_size_deg / 2
            
            # Load data for this grid cell
            tiles = find_tiles_for_bounds(
                tiles_dir, lat, lat + grid_size_deg, lon, lon + grid_size_deg
            )
            
            if not tiles:
                continue
            
            # Load and analyze
            dfs = []
            for tile_name in tiles:
                df = load_tile_data(tiles_dir, tile_name, year)
                if df is not None:
                    mask = (
                        (df['lat'] >= lat) & (df['lat'] < lat + grid_size_deg) &
                        (df['lon'] >= lon) & (df['lon'] < lon + grid_size_deg)
                    )
                    filtered = df[mask]
                    if len(filtered) > 0:
                        dfs.append(filtered)
            
            if not dfs:
                continue
            
            combined = pd.concat(dfs, ignore_index=True)
            
            if len(combined) < MIN_BINS_FOR_CORRELATION * 2:
                continue
            
            # Calculate correlation
            loss = (combined['lossyear_mean'] > 0).astype(float).values
            pop = combined['population_mean'].values
            
            corr, pval, valid = safe_correlation(pop, loss)
            
            if valid and abs(corr) > 0.05:  # Only report meaningful correlations
                hotspots.append({
                    'center_lat': float(center_lat),
                    'center_lon': float(center_lon),
                    'correlation': corr,
                    'p_value': pval,
                    'n_points': len(combined),
                    'loss_rate': float(loss.mean() * 100),
                    'mean_population': float(pop.mean()),
                    'bounds': {
                        'minLat': lat,
                        'maxLat': lat + grid_size_deg,
                        'minLon': lon,
                        'maxLon': lon + grid_size_deg
                    }
                })
                
                print(f"  Found hotspot at ({center_lat:.1f}, {center_lon:.1f}): r={corr:.3f}")
    
    # Sort by correlation strength (absolute value)
    hotspots.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    results = {
        'hotspots': hotspots[:50],  # Top 50
        'positive_correlation_regions': [h for h in hotspots if h['correlation'] > 0][:25],
        'negative_correlation_regions': [h for h in hotspots if h['correlation'] < 0][:25],
        'metadata': {
            'year': year,
            'grid_size_deg': grid_size_deg,
            'total_cells_scanned': len(hotspots),
            'timestamp': datetime.now().isoformat()
        }
    }
    
    os.makedirs(output_dir, exist_ok=True)
    save_json(results, os.path.join(output_dir, 'hotspots.json'))
    
    print(f"\n[Done] Found {len(hotspots)} hotspots")
    print(f"Results saved to {output_dir}/hotspots.json")
    
    return results


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CLI INTERFACE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def main():
    parser = argparse.ArgumentParser(
        description='Forest Loss Spatial Analysis CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Point analysis
  python analysis_cli.py --mode point --lat 50.0 --lon -130.0 --radius 100 --year 2020
  
  # Region analysis
  python analysis_cli.py --mode region --region amazon_west --year 2020
  
  # Global hotspot scan
  python analysis_cli.py --mode hotspots --year 2020
        """
    )
    
    parser.add_argument('--mode', required=True, 
                       choices=['point', 'region', 'hotspots'],
                       help='Analysis mode')
    
    # Point mode arguments
    parser.add_argument('--lat', type=float, help='Center latitude (point mode)')
    parser.add_argument('--lon', type=float, help='Center longitude (point mode)')
    parser.add_argument('--radius', type=float, default=100, 
                       help='Analysis radius in km (point mode)')
    
    # Region mode arguments
    parser.add_argument('--region', help='Region ID (region mode)')
    parser.add_argument('--regions-file', help='Path to regions.json')
    
    # Common arguments
    parser.add_argument('--year', type=int, default=2020, help='Year to analyze')
    parser.add_argument('--tiles-dir', required=True, help='Path to tiles directory')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    
    # Hotspot mode arguments
    parser.add_argument('--grid-size', type=float, default=5.0,
                       help='Grid size in degrees for hotspot scan')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'point':
            if args.lat is None or args.lon is None:
                parser.error("Point mode requires --lat and --lon")
            
            run_point_analysis(
                args.tiles_dir, args.lat, args.lon, 
                args.radius, args.year, args.output_dir
            )
            
        elif args.mode == 'region':
            if args.region is None:
                parser.error("Region mode requires --region")
            if args.regions_file is None:
                parser.error("Region mode requires --regions-file")
            
            run_region_analysis(
                args.tiles_dir, args.region, args.regions_file,
                args.year, args.output_dir
            )
            
        elif args.mode == 'hotspots':
            run_global_hotspot_scan(
                args.tiles_dir, args.year, args.output_dir, 
                args.grid_size
            )
            
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
