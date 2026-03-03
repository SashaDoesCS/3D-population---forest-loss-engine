"""
GLOBAL FOREST CANOPY PIPELINE - WITH LAND FLAG
===============================================

This version adds an 'is_land' column to distinguish:
- Ocean (Hansen NoData) - excluded from output
- Land with no forest (Hansen = 0) - included with is_land=1
- Land with forest (Hansen > 0) - included with is_land=1

This allows the visualization to only load land points,
dramatically reducing file sizes and load times.

Output CSV columns:
- lat, lon: WGS84 coordinates
- canopy_mean: Mean tree canopy cover (0-100%)
- population_count: Total people in this 1km² bin
- population_density: People per km²
- lossyear_mean: Mean loss year value
- loss_fraction: Fraction of pixels with any loss
- is_land: 1 if land, 0 if ocean (ALWAYS 1 in output since we filter)
"""

import os
import time
import gc
import psutil
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window, from_bounds
from numba import jit, prange
from concurrent.futures import ProcessPoolExecutor, as_completed
import urllib.request
import urllib.error
from pyproj import Transformer
import json
from datetime import datetime
import shutil
import warnings

warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

# ===================== CONFIG =====================

TEST_MODE = False
TEST_TILES = ["50N_130W"]

NUM_YEAR_WORKERS = 3
YEARS = [2000, 2005, 2010, 2015, 2020]

# Cleanup settings
CLEANUP_RAW_FILES = False
CLEANUP_INTERMEDIATE = True
KEEP_POPULATION_CACHE = False
MIN_FREE_DISK_GB = 50

BIN_SIZE_METERS = 1000
TARGET_CRS = "EPSG:6933"

ENABLE_DOWNLOAD = True
HANSEN_BASE_URL = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12"

BASE_DATA_DIR = "Data"
CANOPY_RAW_DIR = os.path.join(BASE_DATA_DIR, "Canopy")
CANOPY_PRE_DIR = os.path.join(BASE_DATA_DIR, "Canopy_Preprocessed")
POP_PRE_DIR = os.path.join(BASE_DATA_DIR, "Population_Preprocessed")
OUTPUT_DIR = "output/tiles"
PROGRESS_FILE = "processing_progress.json"

for d in [CANOPY_RAW_DIR, CANOPY_PRE_DIR, POP_PRE_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)


# ===================== TILE GENERATION =====================

def generate_all_tile_names():
    tiles = []
    lat_bands = list(range(80, -1, -10)) + list(range(-10, -60, -10))
    lon_bands = list(range(-180, 180, 10))

    for lat in lat_bands:
        for lon in lon_bands:
            lat_str = f"{abs(lat):02d}{'N' if lat >= 0 else 'S'}"
            lon_str = f"{abs(lon):03d}{'W' if lon < 0 else 'E'}"
            tiles.append(f"{lat_str}_{lon_str}")

    return tiles


def get_tiles_to_process():
    if TEST_MODE:
        return TEST_TILES
    return generate_all_tile_names()


# ===================== PROGRESS TRACKING =====================

class ProgressTracker:
    def __init__(self, filepath=PROGRESS_FILE):
        self.filepath = filepath
        self.state = self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {'completed': [], 'failed': [], 'skipped': [],
                'started_at': datetime.now().isoformat()}

    def _save(self):
        self.state['last_updated'] = datetime.now().isoformat()
        with open(self.filepath, 'w') as f:
            json.dump(self.state, f, indent=2)

    def is_completed(self, tile):
        return tile in self.state['completed']

    def mark_completed(self, tile, stats=None):
        if tile not in self.state['completed']:
            self.state['completed'].append(tile)
        if tile in self.state['failed']:
            self.state['failed'].remove(tile)
        self.state.setdefault('stats', {})[tile] = stats or {}
        self._save()

    def mark_failed(self, tile, error):
        if tile not in self.state['failed']:
            self.state['failed'].append(tile)
        self.state.setdefault('errors', {})[tile] = str(error)[:500]
        self._save()

    def mark_skipped(self, tile, reason):
        if tile not in self.state['skipped']:
            self.state['skipped'].append(tile)
        self._save()

    def get_remaining(self, all_tiles):
        done = set(self.state['completed'] + self.state['skipped'])
        return [t for t in all_tiles if t not in done]

    def summary(self):
        return {
            'completed': len(self.state['completed']),
            'failed': len(self.state['failed']),
            'skipped': len(self.state['skipped'])
        }
    
    def reset(self):
        self.state = {'completed': [], 'failed': [], 'skipped': [],
                      'started_at': datetime.now().isoformat()}
        self._save()


# ===================== UTILITIES =====================

def get_memory_gb():
    return psutil.Process().memory_info().rss / (1024**3)

def get_free_disk_gb(path="."):
    return shutil.disk_usage(path).free / (1024**3)

def aggressive_gc():
    gc.collect()
    gc.collect()

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    return f"{seconds/3600:.1f}hr"

def format_size(bytes_val):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}TB"


# ===================== CLEANUP =====================

def cleanup_tile_files(tile):
    files_removed = 0
    bytes_freed = 0

    if CLEANUP_RAW_FILES:
        for pattern in [f"Hansen_GFC-2024-v1.12_treecover2000_{tile}.tif",
                       f"Hansen_GFC-2024-v1.12_lossyear_{tile}.tif",
                       f"Hansen_GFC-2024-v1.12_datamask_{tile}.tif"]:
            f = os.path.join(CANOPY_RAW_DIR, pattern)
            if os.path.exists(f):
                bytes_freed += os.path.getsize(f)
                os.remove(f)
                files_removed += 1

    if CLEANUP_INTERMEDIATE:
        for pattern in [f"canopy_{tile}_EPSG6933.tif", f"lossyear_{tile}_EPSG6933.tif", f"landmask_{tile}_EPSG6933.tif"]:
            f = os.path.join(CANOPY_PRE_DIR, pattern)
            if os.path.exists(f):
                bytes_freed += os.path.getsize(f)
                os.remove(f)
                files_removed += 1

    if not KEEP_POPULATION_CACHE:
        for year in YEARS:
            f = os.path.join(POP_PRE_DIR, f"gpw_{year}_{tile}_1km_aligned.tif")
            if os.path.exists(f):
                bytes_freed += os.path.getsize(f)
                os.remove(f)
                files_removed += 1

    return files_removed, bytes_freed


# ===================== DOWNLOAD =====================

def download_file(url, dest, retries=3):
    if os.path.exists(dest):
        return True, "exists"

    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest + ".tmp")
            os.rename(dest + ".tmp", dest)
            return True, "downloaded"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False, "not_found"
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
        except:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return False, "failed"


def download_tile_data(tile):
    """
    Download Hansen GFC data for a tile:
    - treecover2000 (canopy)
    - lossyear
    - datamask (land/water mask)
    
    Datamask values:
    - 0 = No data
    - 1 = Land
    - 2 = Water
    """
    canopy_path = os.path.join(CANOPY_RAW_DIR, f"Hansen_GFC-2024-v1.12_treecover2000_{tile}.tif")
    lossyear_path = os.path.join(CANOPY_RAW_DIR, f"Hansen_GFC-2024-v1.12_lossyear_{tile}.tif")
    datamask_path = os.path.join(CANOPY_RAW_DIR, f"Hansen_GFC-2024-v1.12_datamask_{tile}.tif")

    # Check if all files exist
    if os.path.exists(canopy_path) and os.path.exists(lossyear_path) and os.path.exists(datamask_path):
        return canopy_path, lossyear_path, datamask_path, "cached"

    # Download canopy (treecover2000)
    s1, st1 = download_file(f"{HANSEN_BASE_URL}/Hansen_GFC-2024-v1.12_treecover2000_{tile}.tif", canopy_path)
    if not s1:
        return None, None, None, st1

    # Download lossyear
    s2, st2 = download_file(f"{HANSEN_BASE_URL}/Hansen_GFC-2024-v1.12_lossyear_{tile}.tif", lossyear_path)
    if not s2:
        if os.path.exists(canopy_path):
            os.remove(canopy_path)
        return None, None, None, st2

    # Download datamask (land/water)
    s3, st3 = download_file(f"{HANSEN_BASE_URL}/Hansen_GFC-2024-v1.12_datamask_{tile}.tif", datamask_path)
    if not s3:
        # Datamask is essential - cleanup and fail
        if os.path.exists(canopy_path):
            os.remove(canopy_path)
        if os.path.exists(lossyear_path):
            os.remove(lossyear_path)
        return None, None, None, f"datamask_{st3}"

    return canopy_path, lossyear_path, datamask_path, "downloaded"


# ===================== PREPROCESSING =====================

def preprocess_canopy(tile, canopy_raw, lossyear_raw, datamask_raw):
    """
    Reproject canopy, lossyear, and datamask to EPSG:6933.
    
    Uses the official Hansen datamask layer:
    - 0 = No data (not mapped)
    - 1 = Land surface
    - 2 = Permanent water bodies
    
    We create a binary land mask: 1 where datamask == 1 (land), 0 elsewhere
    """
    canopy_out = os.path.join(CANOPY_PRE_DIR, f"canopy_{tile}_EPSG6933.tif")
    lossyear_out = os.path.join(CANOPY_PRE_DIR, f"lossyear_{tile}_EPSG6933.tif")
    landmask_out = os.path.join(CANOPY_PRE_DIR, f"landmask_{tile}_EPSG6933.tif")

    # Check if all outputs exist
    if os.path.exists(canopy_out) and os.path.exists(lossyear_out) and os.path.exists(landmask_out):
        return canopy_out, lossyear_out, landmask_out

    # Create land mask from official datamask
    if not os.path.exists(landmask_out):
        print(f"    Creating land mask from datamask...")
        with rasterio.open(datamask_raw) as src:
            datamask_data = src.read(1)
            
            # Count values for debugging
            n_nodata = np.sum(datamask_data == 0)
            n_land = np.sum(datamask_data == 1)
            n_water = np.sum(datamask_data == 2)
            print(f"    Datamask raw: {n_land:,} land, {n_water:,} water, {n_nodata:,} nodata")
            
            # Create binary land mask: 1 = land (datamask == 1), 0 = water/nodata
            land_mask = (datamask_data == 1).astype(np.uint8)
            
            # Reproject land mask
            transform, width, height = calculate_default_transform(
                src.crs, TARGET_CRS, src.width, src.height, *src.bounds
            )
            
            land_mask_reproj = np.zeros((height, width), dtype=np.uint8)
            reproject(
                source=land_mask,
                destination=land_mask_reproj,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                resampling=Resampling.nearest
            )
            
            # Save land mask
            meta = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": 1,
                "dtype": "uint8",
                "crs": TARGET_CRS,
                "transform": transform,
                "compress": "lzw"
            }
            with rasterio.open(landmask_out, "w", **meta) as dst:
                dst.write(land_mask_reproj, 1)
            
            n_land_reproj = np.sum(land_mask_reproj == 1)
            n_other_reproj = np.sum(land_mask_reproj == 0)
            print(f"    Reprojected land mask: {n_land_reproj:,} land, {n_other_reproj:,} water/nodata")
            
            del datamask_data, land_mask, land_mask_reproj
            gc.collect()

    # Reproject canopy and lossyear
    for src_path, dst_path in [(canopy_raw, canopy_out), (lossyear_raw, lossyear_out)]:
        if os.path.exists(dst_path):
            continue

        with rasterio.open(src_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, TARGET_CRS, src.width, src.height, *src.bounds
            )
            meta = src.meta.copy()
            meta.update({
                "crs": TARGET_CRS, 
                "transform": transform,
                "width": width, 
                "height": height
            })

            with rasterio.open(dst_path, "w", **meta) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=TARGET_CRS,
                    resampling=Resampling.nearest
                )
        aggressive_gc()

    return canopy_out, lossyear_out, landmask_out


def find_gpw_count_file(year):
    """Find GPW population COUNT file for a given year"""
    patterns = [
        os.path.join(BASE_DATA_DIR, "Population", f"gpw_v4_population_count_rev11_{year}_30_sec.tif"),
        os.path.join(BASE_DATA_DIR, "Population", f"gpw-v4-population-count-rev11_{year}_30_sec.tif"),
        os.path.join(BASE_DATA_DIR, "Population", f"gpw_v4_population_count_{year}_30_sec.tif"),
        os.path.join(BASE_DATA_DIR, "Population", f"gpw_{year}_count.tif"),
    ]
    for p in patterns:
        if os.path.exists(p):
            return p
    return None


def preprocess_population(tile, canopy_path, years, force_reprocess=False):
    """Preprocess population from GPW COUNT files."""

    years_to_process = []
    for y in years:
        out_path = os.path.join(POP_PRE_DIR, f"gpw_{y}_{tile}_1km_aligned.tif")
        if force_reprocess or not os.path.exists(out_path):
            years_to_process.append(y)
            if force_reprocess and os.path.exists(out_path):
                os.remove(out_path)

    if not years_to_process:
        return True, {}

    if not os.path.exists(canopy_path):
        print(f"    ❌ Canopy file not found: {canopy_path}")
        return False, {}

    with rasterio.open(canopy_path) as src:
        canopy_bounds = src.bounds
        canopy_crs = src.crs
        canopy_transform = src.transform
        canopy_shape = (src.height, src.width)
        pixel_size_m = abs(canopy_transform.a)

    bin_size_px = int(round(BIN_SIZE_METERS / pixel_size_m))
    n_bins_y = canopy_shape[0] // bin_size_px
    n_bins_x = canopy_shape[1] // bin_size_px
    
    if n_bins_y == 0 or n_bins_x == 0:
        return True, {}

    grid_transform = rasterio.transform.from_bounds(
        canopy_bounds.left, canopy_bounds.bottom,
        canopy_bounds.left + n_bins_x * BIN_SIZE_METERS,
        canopy_bounds.bottom + n_bins_y * BIN_SIZE_METERS,
        n_bins_x, n_bins_y
    )

    pop_totals = {}

    for year in years_to_process:
        out_path = os.path.join(POP_PRE_DIR, f"gpw_{year}_{tile}_1km_aligned.tif")
        if os.path.exists(out_path):
            continue

        gpw_path = find_gpw_count_file(year)
        if not gpw_path:
            continue

        try:
            with rasterio.open(gpw_path) as src:
                from rasterio.warp import transform_bounds
                gpw_bounds_wgs84 = transform_bounds(canopy_crs, src.crs, *canopy_bounds)

                window = from_bounds(*gpw_bounds_wgs84, src.transform)
                col_off = int(round(window.col_off))
                row_off = int(round(window.row_off))
                width = max(1, int(round(window.width)))
                height = max(1, int(round(window.height)))
                window = Window(col_off, row_off, width, height)

                gpw_data = src.read(1, window=window, boundless=True, fill_value=0).astype(np.float64)
                gpw_transform = rasterio.windows.transform(window, src.transform)
                gpw_crs = src.crs
                gpw_nodata = src.nodata

                if gpw_nodata is not None:
                    gpw_data[gpw_data == gpw_nodata] = 0
                gpw_data[gpw_data < 0] = 0

            pop_1km = np.zeros((n_bins_y, n_bins_x), dtype=np.float64)
            reproject(
                source=gpw_data,
                destination=pop_1km,
                src_transform=gpw_transform,
                src_crs=gpw_crs,
                dst_transform=grid_transform,
                dst_crs=canopy_crs,
                resampling=Resampling.sum
            )

            del gpw_data
            aggressive_gc()

            pop_totals[year] = float(np.nansum(pop_1km))

            out_meta = {
                "driver": "GTiff", 
                "height": n_bins_y, 
                "width": n_bins_x,
                "count": 1, 
                "dtype": "float32", 
                "crs": canopy_crs,
                "transform": grid_transform, 
                "compress": "lzw", 
                "nodata": 0
            }

            with rasterio.open(out_path, "w", **out_meta) as dst:
                dst.write(pop_1km.astype(np.float32), 1)

            del pop_1km
            aggressive_gc()

        except Exception as e:
            print(f"    ⚠️ Population {year} failed: {e}")

    return True, pop_totals


# ===================== NUMBA BINNING WITH LAND DETECTION =====================

@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def bin_raster_mean_with_land(data, nodata_val, bin_size):
    """
    Bin raster data and also track which bins have ANY valid (land) pixels.
    
    Returns:
        mean_values: Mean of valid pixels in each bin (NaN if all nodata)
        is_land: 1 if bin has any valid pixels, 0 if all nodata (ocean)
    """
    h, w = data.shape
    ny, nx = h // bin_size, w // bin_size
    out_mean = np.empty((ny, nx), dtype=np.float32)
    out_land = np.empty((ny, nx), dtype=np.uint8)

    for i in prange(ny):
        for j in range(nx):
            total = 0.0
            count = 0
            for bi in range(bin_size):
                for bj in range(bin_size):
                    v = data[i * bin_size + bi, j * bin_size + bj]
                    # Check if valid (not nodata)
                    if v != nodata_val and v >= 0 and v <= 100:
                        total += v
                        count += 1
            
            if count > 0:
                out_mean[i, j] = total / count
                out_land[i, j] = 1  # Has land pixels
            else:
                out_mean[i, j] = np.nan
                out_land[i, j] = 0  # All ocean/nodata
                
    return out_mean, out_land


@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def bin_raster_mean(data, bin_size):
    h, w = data.shape
    ny, nx = h // bin_size, w // bin_size
    out = np.empty((ny, nx), dtype=np.float32)

    for i in prange(ny):
        for j in range(nx):
            total, count = 0.0, 0
            for bi in range(bin_size):
                for bj in range(bin_size):
                    v = data[i * bin_size + bi, j * bin_size + bj]
                    if not np.isnan(v):
                        total += v
                        count += 1
            out[i, j] = total / count if count > 0 else np.nan
    return out


@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def bin_raster_max(data, bin_size):
    """Bin by taking max value - used for land mask (any land pixel = land bin)"""
    h, w = data.shape
    ny, nx = h // bin_size, w // bin_size
    out = np.zeros((ny, nx), dtype=np.uint8)

    for i in prange(ny):
        for j in range(nx):
            max_val = 0
            for bi in range(bin_size):
                for bj in range(bin_size):
                    v = data[i * bin_size + bi, j * bin_size + bj]
                    if v > max_val:
                        max_val = v
            out[i, j] = max_val
    return out


@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def bin_raster_loss_fraction(data, bin_size):
    h, w = data.shape
    ny, nx = h // bin_size, w // bin_size
    out = np.empty((ny, nx), dtype=np.float32)

    for i in prange(ny):
        for j in range(nx):
            loss_count, valid_count = 0, 0
            for bi in range(bin_size):
                for bj in range(bin_size):
                    v = data[i * bin_size + bi, j * bin_size + bj]
                    if not np.isnan(v):
                        valid_count += 1
                        if v > 0:
                            loss_count += 1
            out[i, j] = loss_count / valid_count if valid_count > 0 else np.nan
    return out


# ===================== YEAR PROCESSING WITH LAND FLAG =====================

def process_year_worker(args):
    """
    Process a single year for a tile.
    
    IMPORTANT: Only outputs LAND points using the pre-computed land mask.
    Ocean tiles are completely excluded.
    
    Output CSV columns:
    - lat, lon: WGS84 coordinates
    - canopy_mean: Mean canopy cover (0-100)
    - population_count: Total people in bin
    - population_density: People per km²
    - lossyear_mean: Mean loss year
    - loss_fraction: Fraction of pixels with loss
    """
    year, tile, canopy_path, lossyear_path, landmask_path, out_dir, bin_size_px = args

    start = time.time()
    out_path = os.path.join(out_dir, f"canopy_population_loss_{year}.csv")
    pop_path = os.path.join(POP_PRE_DIR, f"gpw_{year}_{tile}_1km_aligned.tif")

    if os.path.exists(out_path):
        return year, True, 0, "exists"

    if not os.path.exists(pop_path):
        return year, False, 0, "no_population"

    if not os.path.exists(landmask_path):
        return year, False, 0, "no_landmask"

    try:
        # Read land mask
        with rasterio.open(landmask_path) as src:
            landmask_raw = src.read(1)
        
        # Read canopy
        with rasterio.open(canopy_path) as src:
            canopy_shape = (src.height, src.width)
            transform = src.transform
            canopy_raw = src.read(1).astype(np.float32)
            # Replace nodata (255) with 0 for canopy calculation
            canopy_raw[canopy_raw > 100] = 0

        n_bins_y = canopy_shape[0] // bin_size_px
        n_bins_x = canopy_shape[1] // bin_size_px

        # Trim to bin boundaries
        canopy_trimmed = canopy_raw[:n_bins_y * bin_size_px, :n_bins_x * bin_size_px]
        landmask_trimmed = landmask_raw[:n_bins_y * bin_size_px, :n_bins_x * bin_size_px]
        
        # Bin canopy (mean)
        canopy_binned = bin_raster_mean(canopy_trimmed, bin_size_px)
        
        # Bin land mask (any land pixel in bin = land bin)
        # Use max so if ANY pixel in bin is land (1), the bin is land
        landmask_binned = bin_raster_max(landmask_trimmed, bin_size_px)
        is_land = (landmask_binned > 0).astype(np.uint8)
        
        del canopy_raw, canopy_trimmed, landmask_raw, landmask_trimmed
        gc.collect()

        # Bin lossyear
        with rasterio.open(lossyear_path) as src:
            lossyear = src.read(1).astype(np.float32)
        lossyear = lossyear[:n_bins_y * bin_size_px, :n_bins_x * bin_size_px]
        lossyear_mean = bin_raster_mean(lossyear, bin_size_px)
        loss_fraction = bin_raster_loss_fraction(lossyear, bin_size_px)
        del lossyear
        gc.collect()

        # Load population
        with rasterio.open(pop_path) as src:
            pop_1km = src.read(1).astype(np.float32)
        pop_binned = pop_1km[:n_bins_y, :n_bins_x]
        del pop_1km
        gc.collect()

        # Generate coordinates
        pixel_size = abs(transform.a)
        origin_x, origin_y = transform.c, transform.f

        rows, cols = np.mgrid[0:n_bins_y, 0:n_bins_x]
        x_coords = origin_x + (cols * bin_size_px + bin_size_px // 2) * pixel_size
        y_coords = origin_y - (rows * bin_size_px + bin_size_px // 2) * pixel_size

        transformer = Transformer.from_crs("EPSG:6933", "EPSG:4326", always_xy=True)
        lons, lats = transformer.transform(x_coords.ravel(), y_coords.ravel())

        # FILTER: Only include LAND points (is_land == 1)
        land_mask = is_land.ravel() == 1
        n_land = land_mask.sum()
        n_total = len(land_mask)
        n_ocean = n_total - n_land

        if n_land == 0:
            # Entire tile is ocean - write empty CSV
            with open(out_path, 'w') as f:
                f.write("lat,lon,canopy_mean,population_count,population_density,lossyear_mean,loss_fraction\n")
            return year, True, time.time() - start, f"0_land_{n_ocean}_ocean"

        # Calculate bin area
        bin_area_km2 = (BIN_SIZE_METERS / 1000) ** 2

        # Extract only land points
        lats_arr = np.array(lats)[land_mask]
        lons_arr = np.array(lons)[land_mask]
        canopy_arr = canopy_binned.ravel()[land_mask]
        pop_count_arr = pop_binned.ravel()[land_mask]
        pop_density_arr = pop_count_arr / bin_area_km2
        loss_mean_arr = lossyear_mean.ravel()[land_mask]
        loss_frac_arr = loss_fraction.ravel()[land_mask]

        # Replace NaN with 0 for cleaner output
        canopy_arr = np.nan_to_num(canopy_arr, nan=0.0)
        loss_mean_arr = np.nan_to_num(loss_mean_arr, nan=0.0)
        loss_frac_arr = np.nan_to_num(loss_frac_arr, nan=0.0)

        # Write CSV - ONLY land points
        with open(out_path, 'w') as f:
            f.write("lat,lon,canopy_mean,population_count,population_density,lossyear_mean,loss_fraction\n")

            for i in range(n_land):
                f.write(f"{lats_arr[i]:.6f},{lons_arr[i]:.6f},"
                       f"{canopy_arr[i]:.2f},{pop_count_arr[i]:.2f},{pop_density_arr[i]:.2f},"
                       f"{loss_mean_arr[i]:.4f},{loss_frac_arr[i]:.4f}\n")

        return year, True, time.time() - start, f"{n_land}_land_{n_ocean}_ocean"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return year, False, time.time() - start, str(e)[:100]


def process_years(tile, canopy_path, lossyear_path, landmask_path, out_dir, years):
    with rasterio.open(canopy_path) as src:
        pixel_size = abs(src.transform.a)
    bin_size_px = int(round(BIN_SIZE_METERS / pixel_size))

    args_list = [(year, tile, canopy_path, lossyear_path, landmask_path, out_dir, bin_size_px)
                 for year in years]

    results = {}
    with ProcessPoolExecutor(max_workers=NUM_YEAR_WORKERS) as executor:
        futures = {executor.submit(process_year_worker, args): args[0] for args in args_list}
        for future in as_completed(futures):
            year = futures[future]
            try:
                result_year, success, elapsed, info = future.result()
                results[result_year] = {"success": success, "elapsed": elapsed, "info": info}
                print(f"    Year {result_year}: {info}")
            except Exception as e:
                results[year] = {"success": False, "error": str(e)}

    return results


# ===================== MAIN TILE PROCESSING =====================

def process_tile(tile, progress, force_reprocess_pop=False):
    tile_start = time.time()
    stats = {"tile": tile}

    print(f"\n{'='*60}")
    print(f"Processing: {tile}")
    print(f"{'='*60}")

    if get_free_disk_gb() < MIN_FREE_DISK_GB:
        print(f"  ⚠️  Low disk space!")
        return False, stats

    try:
        print("  📥 Downloading (canopy, lossyear, datamask)...")
        canopy_raw, lossyear_raw, datamask_raw, dl_status = download_tile_data(tile)

        if dl_status == "not_found":
            print("  ⏭️  Tile not found (ocean)")
            progress.mark_skipped(tile, "not_found")
            return True, stats

        if canopy_raw is None:
            print(f"  ❌ Download failed: {dl_status}")
            progress.mark_failed(tile, f"download_{dl_status}")
            return False, stats

        print("  🔄 Preprocessing (using official datamask)...")
        canopy_path, lossyear_path, landmask_path = preprocess_canopy(tile, canopy_raw, lossyear_raw, datamask_raw)

        print("  🔄 Preprocessing population...")
        pop_success, pop_totals = preprocess_population(
            tile, canopy_path, YEARS, 
            force_reprocess=force_reprocess_pop
        )
        stats["population_totals"] = pop_totals

        print("  🔄 Processing years (LAND ONLY)...")
        out_dir = os.path.join(OUTPUT_DIR, tile)
        os.makedirs(out_dir, exist_ok=True)

        year_results = process_years(tile, canopy_path, lossyear_path, landmask_path, out_dir, YEARS)
        successful_years = sum(1 for r in year_results.values() if r.get("success"))
        print(f"  ✅ {successful_years}/{len(YEARS)} years")

        if CLEANUP_RAW_FILES or CLEANUP_INTERMEDIATE:
            files, bytes_freed = cleanup_tile_files(tile)
            if bytes_freed > 0:
                print(f"  🧹 Freed {format_size(bytes_freed)}")

        elapsed = time.time() - tile_start
        stats["elapsed"] = elapsed
        print(f"  ✅ Done in {format_time(elapsed)}")

        progress.mark_completed(tile, stats)
        return True, stats

    except Exception as e:
        import traceback
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
        progress.mark_failed(tile, str(e)[:500])
        return False, stats


# ===================== MAIN =====================

def main():
    print(f"\n{'='*60}")
    print("GLOBAL FOREST CANOPY PIPELINE")
    print("LAND-ONLY VERSION (Ocean filtered out)")
    print('='*60)
    
    # Check for GPW files
    missing_years = []
    for year in YEARS:
        if find_gpw_count_file(year) is None:
            missing_years.append(year)
    
    if missing_years:
        print(f"\n⚠️  Missing GPW count files for: {missing_years}")
    else:
        print(f"\n✅ Found GPW count files for all years")
    
    print('='*60)

    all_tiles = get_tiles_to_process()
    progress = ProgressTracker()
    
    remaining = progress.get_remaining(all_tiles)

    summary = progress.summary()
    print(f"\nProgress: {summary['completed']} done, {summary['skipped']} skipped, {len(remaining)} remaining")
    print(f"System: {psutil.virtual_memory().total/(1024**3):.1f}GB RAM, {get_free_disk_gb():.1f}GB disk free")

    if not remaining:
        print("\n✅ All tiles processed!")
        return

    print(f"\n🚀 Processing {len(remaining)} tiles...")

    start_time = time.time()
    success_count = 0

    for i, tile in enumerate(remaining):
        print(f"\n[{i+1}/{len(remaining)}]", end="")
        success, _ = process_tile(tile, progress, force_reprocess_pop=False)
        
        if success:
            success_count += 1

        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed * 3600
        remaining_count = len(remaining) - i - 1
        eta = remaining_count / rate if rate > 0 else 0
        print(f"  Progress: {success_count}/{i+1} | ETA: {format_time(eta * 3600)}")

    print(f"\n{'='*60}")
    print(f"COMPLETE: {success_count}/{len(remaining)} tiles in {format_time(time.time() - start_time)}")
    print('='*60)


if __name__ == "__main__":
    main()