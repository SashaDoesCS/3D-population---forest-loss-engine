"""
DECK.GL DATA PREPARATION - LAND ONLY VERSION
=============================================

This version expects CSVs that already have ocean filtered out.
Creates compact binary files that load quickly.

Expected reduction:
- Raw: ~260M points (~6.8GB per year)
- Land only: ~50-80M points (~1.5-2GB per year)
- Load time: ~5-15 seconds instead of minutes

Binary format per point (20 bytes - optimized):
- lat (float32) - 4 bytes
- lon (float32) - 4 bytes
- canopy (uint8) - 1 byte (0-100)
- pop_density_log (uint8) - 1 byte (log scale 0-255)
- loss_year (uint8) - 1 byte (0-24)
- padding (uint8) - 1 byte (for alignment)
- population_count (float32) - 4 bytes
- loss_fraction (float32) - 4 bytes

Total: 20 bytes per point
"""

import os
import json
import numpy as np
import pandas as pd
import struct
from datetime import datetime
import time

# ===================== CONFIGURATION =====================

TILES_DIR = "output/tiles"
OUTPUT_DIR = "deckgl_data"
YEARS = [2000, 2005, 2010, 2015, 2020]

# Compact format: 20 bytes per point
BYTES_PER_POINT = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)


def discover_tiles(tiles_dir):
    """Find all tile directories"""
    if not os.path.exists(tiles_dir):
        return []

    tiles = []
    for name in sorted(os.listdir(tiles_dir)):
        path = os.path.join(tiles_dir, name)
        if os.path.isdir(path):
            if any(f.endswith('.csv') for f in os.listdir(path)):
                tiles.append(name)
    return tiles


def encode_pop_density_log(density):
    """
    Encode population density to uint8 using log scale.
    0 -> 0
    0.1 -> ~50
    1 -> ~100
    10 -> ~150
    100 -> ~200
    1000+ -> 255
    """
    if density <= 0:
        return 0
    # Log scale: log10(density) from -1 to 3 maps to 0-255
    log_val = np.log10(np.maximum(density, 0.1))  # -1 to 3+
    scaled = ((log_val + 1) / 4) * 255  # Map to 0-255
    return np.clip(scaled, 0, 255).astype(np.uint8)


def process_year(year, tiles_dir, output_dir, all_tiles):
    """Process one year - create single compact binary file"""

    print(f"\n{'='*50}")
    print(f"YEAR {year}")
    print(f"{'='*50}")

    bin_path = os.path.join(output_dir, f"land_{year}.bin")
    index_path = os.path.join(output_dir, f"land_{year}_index.json")

    start_time = time.time()

    # First pass: count total land points
    print("Pass 1: Counting land points...")
    tile_info = {}
    total_points = 0

    for tile_name in all_tiles:
        csv_path = os.path.join(tiles_dir, tile_name, f"canopy_population_loss_{year}.csv")
        if not os.path.exists(csv_path):
            continue

        try:
            # Just count rows (skip header)
            with open(csv_path, 'r') as f:
                count = sum(1 for _ in f) - 1  # Subtract header
            
            if count > 0:
                tile_info[tile_name] = {'count': count}
                total_points += count
        except Exception as e:
            print(f"  âš  {tile_name}: {e}")

    if total_points == 0:
        print(f"  No data for {year}")
        return None

    print(f"  Total land points: {total_points:,} across {len(tile_info)} tiles")
    estimated_size = total_points * BYTES_PER_POINT / 1024 / 1024
    print(f"  Estimated file size: {estimated_size:.1f} MB")

    # Second pass: write binary
    print("Pass 2: Writing compact binary...")

    total_population = 0
    tiles_written = 0
    points_written = 0

    with open(bin_path, 'wb') as f:
        # Write header: total points (4 bytes)
        f.write(struct.pack('I', total_points))

        for tile_name, info in tile_info.items():
            csv_path = os.path.join(tiles_dir, tile_name, f"canopy_population_loss_{year}.csv")

            try:
                df = pd.read_csv(csv_path)
                
                if len(df) == 0:
                    continue

                # Record byte offset for this tile
                info['byte_offset'] = 4 + points_written * BYTES_PER_POINT
                info['points'] = len(df)

                # Get bounds for spatial indexing
                info['bounds'] = {
                    'minLat': float(df['lat'].min()),
                    'maxLat': float(df['lat'].max()),
                    'minLon': float(df['lon'].min()),
                    'maxLon': float(df['lon'].max()),
                }

                # Write each point in compact format
                for _, row in df.iterrows():
                    lat = float(row['lat'])
                    lon = float(row['lon'])
                    canopy = int(np.clip(row['canopy_mean'], 0, 100))
                    pop_density = float(row.get('population_density', row.get('population_mean', 0)))
                    pop_count = float(row.get('population_count', row.get('population_mean', 0)))
                    loss_year = int(np.clip(row['lossyear_mean'], 0, 24))
                    loss_frac = float(row.get('loss_fraction', 0))

                    # Encode density to log scale uint8
                    pop_density_log = encode_pop_density_log(pop_density)

                    # Pack: lat(f), lon(f), canopy(B), pop_log(B), loss_year(B), pad(B), pop_count(f), loss_frac(f)
                    f.write(struct.pack('ff BBBB ff',
                        lat, lon,
                        canopy, pop_density_log, loss_year, 0,  # 4 bytes padding to align
                        pop_count, loss_frac
                    ))

                    total_population += pop_count
                    points_written += 1

                tiles_written += 1

                if tiles_written % 50 == 0:
                    print(f"  Written {tiles_written}/{len(tile_info)} tiles ({points_written:,} points)...")

            except Exception as e:
                print(f"  âš  Error in {tile_name}: {e}")
                import traceback
                traceback.print_exc()

    # Write index
    index_data = {
        'year': year,
        'total_points': total_points,
        'total_population': total_population,
        'bytes_per_point': BYTES_PER_POINT,
        'format': 'compact_land_only',
        'tiles': tile_info
    }

    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=2)

    elapsed = time.time() - start_time
    file_size = os.path.getsize(bin_path)

    print(f"\nâœ… {year} complete:")
    print(f"   Tiles: {tiles_written}")
    print(f"   Points: {points_written:,}")
    print(f"   Population: {total_population:,.0f}")
    print(f"   File size: {file_size / 1024 / 1024:.1f} MB")
    print(f"   Time: {elapsed:.1f}s")

    return {
        'tiles': tiles_written,
        'points': points_written,
        'population': total_population,
        'size': file_size
    }


def main():
    print("\n" + "="*50)
    print("DECK.GL DATA PREP - LAND ONLY")
    print("Creates compact binary files for fast loading")
    print("="*50)

    total_start = time.time()

    # Discover tiles
    all_tiles = discover_tiles(TILES_DIR)
    print(f"\nFound {len(all_tiles)} tile folders")

    if not all_tiles:
        print("âŒ No tiles found!")
        return

    # Process each year
    year_stats = {}
    for year in YEARS:
        stats = process_year(year, TILES_DIR, OUTPUT_DIR, all_tiles)
        if stats:
            year_stats[year] = stats

    # Write manifest
    manifest = {
        'years': list(year_stats.keys()),
        'format': 'compact_land_only',
        'bytes_per_point': BYTES_PER_POINT,
        'fields': ['lat', 'lon', 'canopy', 'pop_density_log', 'loss_year', 'pad', 'pop_count', 'loss_frac'],
        'stats': year_stats,
        'created': datetime.now().isoformat()
    }

    with open(os.path.join(OUTPUT_DIR, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    total_elapsed = time.time() - total_start

    print("\n" + "="*50)
    print("COMPLETE")
    print("="*50)

    total_points = sum(s['points'] for s in year_stats.values())
    total_size = sum(s['size'] for s in year_stats.values())

    print(f"Total points: {total_points:,}")
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB ({total_size / 1024 / 1024 / 1024:.2f} GB)")
    print(f"Time: {total_elapsed:.1f}s")

    print(f"\nOutput files:")
    for year in year_stats:
        print(f"  land_{year}.bin")
    print(f"  manifest.json")


if __name__ == "__main__":
    main()
