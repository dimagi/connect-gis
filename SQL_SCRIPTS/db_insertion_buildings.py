import subprocess
import os
from math import ceil
import geopandas as gpd
from sqlalchemy import create_engine
from shapely.geometry import MultiPolygon
import pandas as pd
from sqlalchemy.sql import text
import json
import argparse
import time
import csv
from datetime import datetime

os.environ["AWS_REQUEST_PAYER"] = "requester"
os.environ["AWS_DEFAULT_REGION"] = "us-west-2"

SUCCESS_LOG_CSV = "successful_tiles.csv"
FAILED_TILES_CSV = "failed_tiles.csv"
MAX_RETRIES = 6

# Nigeria bounding box
left, bottom = 2.6917, 4.2406  # SW corner
right, top = 14.6789, 13.8656  # NE corner

# Configuration
tile_size = 0.3  # degrees
base_filename = "nigeria_buildings"
data_type = "building"
output_format = "geojson"

# Calculate total tiles
num_tiles_x = ceil((right - left) / tile_size)
num_tiles_y = ceil((top - bottom) / tile_size)
total_tiles = num_tiles_x * num_tiles_y

# Database setup
db_connection_string = "postgresql://username:password@localhost:5432/your_database"
engine = create_engine(db_connection_string)


def get_tiles_for_instance(instance_id, total_instances):
    """Process full columns at a time"""
    # Assign full columns to instances
    columns_per_instance = ceil(num_tiles_x / total_instances)
    start_col = instance_id * columns_per_instance
    end_col = min(start_col + columns_per_instance, num_tiles_x)

    tiles = []
    for i in range(start_col, end_col):
        for j in range(num_tiles_y):
            if not is_tile_processed(i, j):
                tiles.append((i, j))
    return tiles


def init_log_files():
    """Initialize both log files with headers if they don't exist"""
    for filename, headers in [
        (SUCCESS_LOG_CSV, ['timestamp', 'instance_id', 'tile_i', 'tile_j']),
        (FAILED_TILES_CSV, [
            'timestamp', 'instance_id', 'tile_i', 'tile_j',
            'tile_size', 'minx', 'miny', 'maxx', 'maxy',
            'attempts', 'last_error'
        ])
    ]:
        if not os.path.exists(filename):
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)


def log_failed_tile(instance_id, i, j, bbox, attempts, error_msg):
    """Log failed tiles to CSV"""
    with open(FAILED_TILES_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            instance_id,
            i,
            j,
            tile_size,
            bbox[0],
            bbox[1],
            bbox[2],
            bbox[3],
            attempts,
            error_msg[:500]  # Truncate long error messages
        ])


def log_successful_tile(instance_id, i, j):
    """Log successfully processed tiles"""
    with open(SUCCESS_LOG_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            instance_id,
            i,
            j
        ])


def is_tile_processed(i, j):
    """Check if tile is already in success log"""
    if not os.path.exists(SUCCESS_LOG_CSV):
        return False
    with open(SUCCESS_LOG_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['tile_i']) == i and int(row['tile_j']) == j:
                return True
    return False


def download_tile(bbox, filename, instance_id, i, j):
    """Download with enhanced retry and logging"""
    last_error = "Unknown error"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = subprocess.run(
                ["overturemaps", "download",
                 f"--bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                 f"-f{output_format}",
                 f"--type={data_type}",
                 f"-o{filename}"],
                check=True,
                capture_output=True,
                text=True,
                timeout=300
            )
            return True
        except subprocess.TimeoutExpired as e:
            last_error = f"Timeout (attempt {attempt})"
        except subprocess.CalledProcessError as e:
            last_error = f"CLI Error: {e.stderr.strip() or 'No error message'}"
        except Exception as e:
            last_error = f"Unexpected error: {str(e)}"

        print(f"Attempt {attempt} failed for tile {i},{j}: {last_error}")
        time.sleep(5 * attempt)  # Exponential backoff

    # If we get here, all attempts failed
    log_failed_tile(instance_id, i, j, bbox, MAX_RETRIES, last_error)
    return False


def process_tile(instance_id, i, j):
    """Process a single tile end-to-end"""

    """Process a tile with resume capability"""
    if is_tile_processed(i, j):
        print(f"Tile {i},{j} already processed, skipping")
        return True

    # Calculate bounding box
    tile_left = left + i * tile_size
    tile_bottom = bottom + j * tile_size
    tile_right = min(tile_left + tile_size, right)
    tile_top = min(tile_bottom + tile_size, top)
    bbox = (tile_left, tile_bottom, tile_right, tile_top)

    # Download
    filename = f"{base_filename}_tile_{i}_{j}.{output_format}"
    print(f"[Tile {i},{j}] Downloading {filename}...")
    if not download_tile(bbox, filename, instance_id, i, j):
        print(f"Permanent failure for tile {i},{j} after {MAX_RETRIES} attempts")
        return False

    # Process
    try:
        print(f"[Tile {i},{j}] Processing data...")
        gdf = gpd.read_file(filename)

        if gdf.empty:
            print(f"[Tile {i},{j}] Empty data, skipping")
            log_successful_tile(instance_id, i, j)
            return True

        # Transform geometries
        gdf['geometry'] = gdf.geometry.apply(
            lambda g: MultiPolygon([g]) if g.geom_type == 'Polygon' else g
        )

        # Process sources
        def parse_sources(s):
            try:
                return json.loads(s) if isinstance(s, str) else (s if isinstance(s, list) else [])
            except json.JSONDecodeError:
                return []

        gdf['sources'] = gdf.sources.apply(parse_sources)

        # Extract metadata
        gdf[['confidence', 'record_id']] = gdf.sources.apply(
            lambda s: (s[0].get('confidence', 0.5), s[0].get('record_id')) if s else (0.5, None)
        ).apply(pd.Series)

        # Calculate spatial metrics
        projected = gdf.to_crs(epsg=3857)
        gdf['area_in_meters'] = projected.geometry.area
        centroids = projected.geometry.centroid.to_crs(epsg=4326)
        gdf['latitude'] = centroids.y
        gdf['longitude'] = centroids.x
        gdf['centroid_geometry'] = centroids

        # Insert to database
        gdf[['latitude', 'longitude', 'area_in_meters', 'confidence',
             'record_id', 'geometry', 'centroid_geometry']].to_postgis(
            'buildings', engine, if_exists='append', index=False)

        print(f"[Tile {i},{j}] Successfully processed {len(gdf)} features")
        log_successful_tile(instance_id, i, j)
        return True

    except Exception as e:
        print(f"[Tile {i},{j}] Processing failed: {str(e)}")
        log_failed_tile(instance_id, i, j, bbox, 1, f"Processing error: {str(e)}")
        return False
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def main():
    init_log_files()
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-id", type=int, required=True, help="Instance ID (0-based)")
    parser.add_argument("--total-instances", type=int, required=True, help="Total number of instances")
    args = parser.parse_args()

    print(f"Instance {args.instance_id} of {args.total_instances} starting - checking for unprocessed tiles")

    tiles = get_tiles_for_instance(args.instance_id, args.total_instances)
    print(f"Found {len(tiles)} tiles to process")

    processed = 0
    for i, j in tiles:
        success = process_tile(args.instance_id, i, j)
        print(f"Tile {i},{j} {'succeeded' if success else 'failed'}")
        processed += 1

    # Index creation by first instance
    if args.instance_id == 0:
        print("Waiting 60s for other instances to finish...")
        time.sleep(60)
        create_indexes()

    print(f"=== Instance {args.instance_id} complete. Processed {processed} tiles ===")


def create_indexes():
    """Create database indexes after all data is loaded"""
    indexes = [
        "CREATE INDEX IF NOT EXISTS buildings_geometry_gist ON buildings USING GIST (geometry)",
        "CREATE INDEX IF NOT EXISTS buildings_centroid_gist ON buildings USING GIST (centroid_geometry)",
        "CREATE INDEX IF NOT EXISTS buildings_location_idx ON buildings (latitude, longitude)",
        "CREATE INDEX IF NOT EXISTS buildings_area_idx ON buildings (area_in_meters)"
    ]

    print("Creating indexes...")
    with engine.begin() as conn:
        for sql in indexes:
            try:
                conn.execute(text(sql))
            except Exception as e:
                print(f"Index creation failed: {str(e)}")
    print("Index creation complete")


if __name__ == "__main__":
    main()
