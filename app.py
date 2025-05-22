import csv
import json
import math
import os
import time
from collections import Counter
from io import StringIO

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
from k_means_constrained import KMeansConstrained
from shapely.geometry import Polygon
from sqlalchemy import create_engine
from sqlalchemy.sql import text

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
load_dotenv()

# Fetch GEE credentials from environment variable
GEE_CREDS = os.getenv("GEE_CREDS", "{}")
GEE_PROJECT_NAME = os.getenv("GEE_PROJECT_NAME", "")
try:
    # Ensure the directory exists
    os.makedirs(os.path.expanduser("~/.config/earthengine"), exist_ok=True)

    # Write credentials from environment variable
    credentials_path = os.path.expanduser("~/.config/earthengine/credentials")
    if not os.path.exists(credentials_path):
        with open(credentials_path, "w") as f:
            f.write(GEE_CREDS)
    ee.Initialize(
        project= GEE_PROJECT_NAME,
        opt_url='https://earthengine-highvolume.googleapis.com'
    )
    buildings = ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons")
except Exception as e:
    print(f"Error initializing Earth Engine: {e}")

# Fetch database connection parameters from environment variables
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "pwd")
DB_HOST = os.getenv("DB_HOST", "host")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "db")

DB_CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DB_CONNECTION_STRING)

@app.route("/")
def home():
    host_url = os.getenv('HOST_URL', "https://connectgis.dimagi.com")
    return render_template("index.html", host_url=host_url)


def _calculate_cluster_sizes(n_samples, num_clusters, balance_tolerance=0.05):
    base_size = math.ceil(n_samples / num_clusters)

    if balance_tolerance == 0:
        min_size = math.floor(n_samples / num_clusters)
        max_size = base_size
    else:
        min_size = base_size if num_clusters == 1 else math.floor(base_size * (1 - balance_tolerance))
        max_size = base_size if num_clusters == 1 else math.ceil(base_size * (1 + balance_tolerance))

    return base_size, min_size, max_size


def _run_constrained_kmeans(coords, num_clusters, min_size, max_size):
    constrained_kmeans = KMeansConstrained(
        n_clusters=num_clusters,
        size_min=min_size,
        size_max=max_size,
        max_iter=300,
        random_state=42,
        n_init=1,
        n_jobs=1
    )
    return constrained_kmeans.fit_predict(coords)


def optimized_balanced_kmeans_constrained_with_buildings_count(buildingsGDF, coords, buildings_per_cluster,
                                                               balance_tolerance=0.05):
    """
    Perform balanced K-means clustering targeting a fixed number of buildings per cluster.

    Parameters:
    - buildingsGDF: GeoDataFrame containing building data
    - coords: numpy array of coordinates (n_samples, 2)
    - buildings_per_cluster: int, target number of buildings per cluster
    - balance_tolerance: float, tolerance for cluster size variation (default: 0.05)

    Returns:
    - GeoDataFrame with cluster labels
    """
    n_samples = len(coords)
    num_clusters = math.ceil(n_samples / buildings_per_cluster)
    _, min_size, max_size = _calculate_cluster_sizes(n_samples, num_clusters, balance_tolerance)

    labels = _run_constrained_kmeans(coords, num_clusters, min_size, max_size)
    return getClustersData(buildingsGDF, coords, labels)


def optimized_balanced_kmeans_constrained_with_no_of_clusters(buildingsGDF, coords, num_clusters=3,
                                                              balance_tolerance=0.05):
    """
    Perform balanced K-means clustering with a specified number of clusters.

    Parameters:
    - buildingsGDF: GeoDataFrame containing building data
    - coords: numpy array of coordinates (n_samples, 2)
    - num_clusters: int, number of clusters (default: 3)
    - balance_tolerance: float, tolerance for cluster size variation (default: 0.05)

    Returns:
    - GeoDataFrame with cluster labels
    """
    n_samples = len(coords)
    _, min_size, max_size = _calculate_cluster_sizes(n_samples, num_clusters, balance_tolerance)

    labels = _run_constrained_kmeans(coords, num_clusters, min_size, max_size)
    return getClustersData(buildingsGDF, coords, labels)

def getClustersData(buildingsGDF, coords, cluster_labels):
    cluster_counts = Counter(cluster_labels)
    # Convert the coordinates and their cluster labels into a list of dictionaries
    clusters = [
        {
            "coordinates": [coord[0], coord[1]],  # Use latitude and longitude directly from the coord tuple
            "cluster": int(cluster_label),  # Assign the cluster label
            "numOfBuildings": cluster_counts[cluster_label]
        }
        for coord, cluster_label in zip(coords, cluster_labels)  # Iterate over coords and cluster labels
    ]
    buildingsGDF["cluster_label"] = cluster_labels.tolist()

    return clusters

def getBuildingsDataFromGEE(polygon_coords, buildings_area_in_meters=0, buildings_confidence=0):
    if not polygon_coords:
        return jsonify({"error": "Invalid polygon coordinates"}), 400

    # Create EE Polygon geometry
    region = ee.Geometry.Polygon(polygon_coords)
    # Filter buildings inside the polygon
    filtered_buildings = buildings.filterBounds(region)
    if buildings_area_in_meters > 0:
        filtered_buildings.filter(ee.Filter.gt('area_in_meters', buildings_area_in_meters))
    if buildings_confidence > 0:
        filtered_buildings.filter(ee.Filter.gt('confidence', buildings_confidence))

    try:
        buildings_geojson = filtered_buildings.getInfo()
    except ee.EEException as e:
        raise Exception(
            "Too many elements: The area contains more than 5000 buildings or grid cells. Please reduce the polygon size.")
    return buildings_geojson

def getBuildingsDataFromDB(polygon_coords, buildings_area_in_meters=0.0, buildings_confidence=0.0):
    """
        Fetch buildings from the building table within the specified polygon.

    Args:
        polygon_coords (list): List of [lng, lat] coordinates defining the polygon.
        buildings_area_in_meters (float): Minimum building area to filter by (only applied if > 0)
        buildings_confidence (float): Minimum confidence score to filter by (only applied if > 0)

        Returns:
            dict: GeoJSON FeatureCollection with building features.
        """
    if not polygon_coords:
        raise ValueError("Invalid polygon coordinates")

    polygon = Polygon(polygon_coords)
    polygon_wkt = polygon.wkt

    # Base query without filters
    query = """
            SELECT DISTINCT ON (latitude, longitude) \
                id, \
                   latitude, \
                   longitude, \
                   area_in_meters, \
                   confidence, \
                   record_id, \
                   ST_AsGeoJSON(ST_GeometryN(geometry, 1)) as geometry
            FROM buildings
            WHERE ST_Within(geometry, ST_GeomFromText(%s, 4326)) \
            """

    # Add filters only if values are greater than 0
    params = [polygon_wkt]

    if buildings_area_in_meters > 0:
        query += " AND area_in_meters >= %s"
        params.append(buildings_area_in_meters)

    if buildings_confidence > 0:
        query += " AND confidence >= %s"
        params.append(buildings_confidence)

    query += ";"  # Add final semicolon

    conn = None
    cur = None
    try:
        conn = engine.raw_connection()
        cur = conn.cursor()
        cur.execute(query, tuple(params))

        # Build features incrementally
        features = []
        for row in cur:
            id, latitude, longitude, area_in_meters, confidence, record_id, geom_json = row
            geometry = json.loads(geom_json)

            feature = {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "id": str(id),
                    "area_in_meters": area_in_meters,
                    "confidence": confidence if confidence is not None else 0,
                    "full_plus_code": record_id,
                    "longitude_latitude": {
                        "type": "Point",
                        "coordinates": [longitude, latitude]
                    }
                }
            }
            features.append(feature)

        # Return as a GeoJSON FeatureCollection dict
        return {
            "type": "FeatureCollection",
            "features": features
        }

    except Exception as e:
        print(f"Error fetching data: {e}")
        return {
            "type": "FeatureCollection",
            "features": []
        }

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

@app.route('/get_building_density', methods=['POST'])
def get_building_density():
    try:
        data = request.get_json()
        clustering_type = data.get("clusteringType")
        no_of_clusters = int(data.get("noOfClusters", 3))
        no_of_buildings = int(data.get("noOfBuildings", 250))
        buildings_area_in_meters = float(data.get("buildingsAreaInMeters", 0))
        buildings_confidence = float(data.get("buildingsConfidence", 0)) / 100
        tolerance = float(data.get("thresholdVal", 10)) / 100  # Percentage to decimal
        fetchClusters = bool(data.get("fetchClusters", False))
        dbType = data.get("dbType")

        if clustering_type == "bottomUp":
            result = handle_bottom_up_clustering(data, no_of_clusters, no_of_buildings, tolerance)
        else:
            result = handle_polygon_based_clustering(data, clustering_type, no_of_clusters, no_of_buildings, tolerance,
                                                     fetchClusters, dbType, buildings_area_in_meters, buildings_confidence)
        return result
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def getBuildingsAroundPin(pin_lng, pin_lat, total_buildings_needed, tolerance=0.1):
    """
    Fetch buildings around a pin with initial 2km box, increment by 1.5x, capped at 10 km,
    then perform a binary search for the optimal range.

    Args:
        pin_lng (float): Longitude of the pin
        pin_lat (float): Latitude of the pin
        total_buildings_needed (int): Desired number of buildings (noOfClusters * noOfBuildings)
        tolerance (float): Tolerance percentage (e.g., 0.1 for ±10%)

    Returns:
        buildings_geojson, coordinates: GeoJSON and numpy array of coordinates
    """
    pin_point = ee.Geometry.Point(pin_lng, pin_lat)

    # Calculate acceptable range with tolerance
    min_buildings = int(total_buildings_needed * (1 - tolerance))
    max_buildings = int(total_buildings_needed * (1 + tolerance))

    # Initial bounding box: 2 km (~0.018 degrees), capped at 10 km
    earth_radius_km = 6371
    km_to_deg = lambda km, lat: km / (earth_radius_km * math.cos(math.radians(lat)) * 2 * math.pi / 360)

    initial_delta = km_to_deg(2, pin_lat)  # 2 km radius in degrees
    max_delta = km_to_deg(10, pin_lat)  # Cap at 10 km radius in degrees

    delta = initial_delta
    bbox = create_bbox(pin_lng, pin_lat, delta)

    # Step 1: Incrementally increase until sufficient buildings or max delta reached
    filtered_buildings = fetch_buildings(bbox, pin_point)
    building_count = filtered_buildings.size().getInfo()

    while building_count < min_buildings and delta < max_delta:
        delta = min(delta * 1.5, max_delta)  # Increase by 1.5x, but do not exceed 10 km
        bbox = create_bbox(pin_lng, pin_lat, delta)
        filtered_buildings = fetch_buildings(bbox, pin_point)
        building_count = filtered_buildings.size().getInfo()

    # If still insufficient, use all available buildings
    if building_count < min_buildings:
        sorted_buildings = filtered_buildings.sort('distance')
    else:
        # Step 2: Binary search to find the smallest sufficient bounding box
        left, right = initial_delta, delta
        precision = km_to_deg(0.1, pin_lat)  # 0.1 km precision

        while right - left > precision:
            mid = (left + right) / 2
            bbox = create_bbox(pin_lng, pin_lat, mid)
            filtered_buildings = fetch_buildings(bbox, pin_point)
            building_count = filtered_buildings.size().getInfo()

            if building_count >= min_buildings:
                right = mid
            else:
                left = mid

        # Final fetch with optimized delta
        delta = right
        bbox = create_bbox(pin_lng, pin_lat, delta)
        filtered_buildings = fetch_buildings(bbox, pin_point)
        sorted_buildings = filtered_buildings.sort('distance').limit(max_buildings)

    try:
        buildings_geojson = sorted_buildings.getInfo()
    except ee.EEException as e:
        raise Exception("Error fetching buildings: Try a smaller area or fewer buildings.")

    coordinates = [feature['properties']['longitude_latitude']['coordinates']
                   for feature in buildings_geojson['features']]

    # Warnings for building counts
    actual_count = len(coordinates)
    if actual_count < min_buildings:
        print(f"Warning: Only {actual_count} buildings found near pin; "
              f"at least {min_buildings} required within tolerance.")

    # Trim excess buildings
    if actual_count > max_buildings:
        coordinates = coordinates[:total_buildings_needed]
        buildings_geojson['features'] = buildings_geojson['features'][:total_buildings_needed]
    elif actual_count < total_buildings_needed and actual_count >= min_buildings:
        print(f"Warning: Fetched {actual_count} buildings, less than {total_buildings_needed} but within tolerance.")

    return buildings_geojson, np.array(coordinates)

def create_bbox(pin_lng, pin_lat, delta):
    """Utility function to create a bounding box around a pin."""
    min_lng, max_lng = pin_lng - delta, pin_lng + delta
    min_lat, max_lat = pin_lat - delta, pin_lat + delta
    return ee.Geometry.Rectangle([min_lng, min_lat, max_lng, max_lat])

def fetch_buildings(bbox, pin_point):
    """Fetch buildings within a bounding box and calculate distances."""
    return buildings.filterBounds(bbox).map(
        lambda feature: feature.set('distance', feature.geometry().centroid().distance(pin_point))
    )

def handle_bottom_up_clustering(data, no_of_clusters, no_of_buildings, tolerance):
    pin = data.get("pin")
    if not pin or len(pin) != 2:
        return jsonify({"error": "Invalid pin coordinates"}), 400

    pin_lng, pin_lat = map(float, pin)
    total_buildings_needed = no_of_clusters * no_of_buildings

    # Fetch buildings within tolerance
    buildings_geojson, coordinates = getBuildingsAroundPin(pin_lng, pin_lat, total_buildings_needed, 0)
    buildings_count = len(coordinates)

    # Validate building count against the tolerance
    min_buildings = int(total_buildings_needed * (1 - tolerance))
    if buildings_count < min_buildings:
        return jsonify({
            "error": f"Insufficient buildings ({buildings_count}) found; at least {min_buildings} required"
        }), 400

    # Perform clustering
    clusters = optimized_balanced_kmeans_constrained_with_no_of_clusters(
        buildings_geojson, coordinates, no_of_clusters, tolerance
    )

    return jsonify({
        "building_count": buildings_count,
        "buildings": buildings_geojson,
        "clusters": clusters
    })


def handle_polygon_based_clustering(data, clustering_type, no_of_clusters, no_of_buildings, tolerance, fetchClusters,
                                    dbType, buildings_area_in_meters, buildings_confidence,):
    polygon_coords = data.get("polygon", [])
    if not polygon_coords:
        return jsonify({"error": "Invalid polygon coordinates"}), 400
    if dbType == 'GEE':
        buildings_geojson = getBuildingsDataFromGEE(polygon_coords)
    else:
        try:
            buildings_geojson = getBuildingsDataFromDB(polygon_coords, buildings_area_in_meters, buildings_confidence)
        except Exception as e:
            return jsonify({"error": f"Error fetching buildings from database: {str(e)}"}), 500

    coordinates = [
        feature['properties']['longitude_latitude']['coordinates']
        for feature in buildings_geojson['features']
    ]
    if not coordinates:
        return jsonify({"message": "No buildings found within the polygon", "building_count": 0}), 404

    coordinates = np.array(coordinates)
    buildings_count = len(coordinates)
    clusters = None

    if (fetchClusters):
        # Perform appropriate clustering based on the type
        if clustering_type == 'kMeans':
            clusters = optimized_balanced_kmeans_constrained_with_no_of_clusters(
                buildings_geojson, coordinates, no_of_clusters, tolerance
            )
        elif clustering_type == 'balancedKMeans':
            clusters = optimized_balanced_kmeans_constrained_with_buildings_count(
                buildings_geojson, coordinates, no_of_buildings, tolerance
            )
        else:
            return jsonify({"error": f"Unsupported clustering type: {clustering_type}"}), 400

    return jsonify({
        "building_count": buildings_count,
        "buildings": buildings_geojson,
        "clusters": clusters
    })

@app.route('/generate_report', methods=['POST'])
def getReports():
    start_time = time.time()
    print('Report generation started')
    report = getReports_sqlQuery()
    return report

def getReports_sqlQuery():
    try:
        requestData = request.get_json()
        data = requestData.get('data')
        fetchVisitToBuildingsVal = bool(requestData.get("fetchVisitToBuildingsVal", True))

        if not data or not isinstance(data, list):
            return jsonify({"error": "No valid data provided"}), 400

        # Validate and extract visit points
        visit_points = []
        for entry in data:
            lat = entry.get('latitude')
            lng = entry.get('longitude')
            flw_id = entry.get('flw_id', -1)
            if lat is not None and lng is not None:
                try:
                    lat = float(lat)
                    lng = float(lng)
                    if -90 <= lat <= 90 and -180 <= lng <= 180:
                        visit_points.append(f"{lng} {lat} {flw_id}")
                    else:
                        print(f"Skipping invalid coordinates: lat={lat}, lng={lng}")
                except (ValueError, TypeError):
                    print(f"Skipping invalid numeric value for lat={lat}, lng={lng}")

        if not visit_points:
            return jsonify({"error": "No valid latitude/longitude data provided"}), 400

        output = StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)
        query = f""
        if not fetchVisitToBuildingsVal:
            query = """ SELECT * FROM public.get_ward_visit_summary(:visit_points) """
            writer.writerow([
                "state_name", "lga_name", "ward_name", "population", "total.visits", "total.buildings",
                "num.phc.serve.ward", "median.visit.to.phc", "max.visit.to.phc", "median.building.to.phc",
                "max.buildings.to.phc", "unique.flws", "coverage"
            ])
        else:
            query = """ SELECT * FROM public.get_ward_visit_summary_with_visit_to_buildings(:visit_points) """
            writer.writerow([
                "state_name", "lga_name", "ward_name", "population", "total.visits", "total.buildings",
                "num.phc.serve.ward", "median.visit.to.phc", "max.visit.to.phc", "median.building.to.phc",
                "max.buildings.to.phc", "unique.flws", "coverage", "percent.building.100.plus.to.visit",
                "percent.building.200.plus.to.visit", "percent.building.500.plus.to.visit","percent.building.10000.plus.to.visit"
            ])
        params = {'visit_points': ','.join(visit_points)}

        # Execute query and fetch results
        with engine.connect() as connection:
            result = connection.execute(text(query), params).fetchall()

        # Generate CSV
        writer.writerows(result)

        # Return CSV response
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=Ward_summary_report.csv"}
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


def generate_grid(polygon_coords, grid_size_meters=50):
    """
    Generate a 50x50 meter grid over the polygon.
    Args:
        polygon_coords: List of [lng, lat] coordinates
        grid_size_meters: Grid size in meters
    Returns:
        GeoDataFrame with grid polygons and centroids
    """
    # Validate polygon coordinates
    if not polygon_coords or len(polygon_coords) < 3:
        raise ValueError("Invalid polygon coordinates: must have at least 3 points")

    try:
        polygon = Polygon(polygon_coords)
        if not polygon.is_valid:
            raise ValueError("Invalid polygon: geometry is not valid")
    except Exception as e:
        raise ValueError(f"Failed to create polygon: {str(e)}")

    avg_lat = polygon.centroid.y
    meters_to_deg = grid_size_meters / (111000 * math.cos(math.radians(avg_lat)))

    minx, miny, maxx, maxy = polygon.bounds
    x_steps = math.ceil((maxx - minx) / meters_to_deg)
    y_steps = math.ceil((maxy - miny) / meters_to_deg)

    grids = []
    for i in range(x_steps):
        for j in range(y_steps):
            x = minx + i * meters_to_deg
            y = miny + j * meters_to_deg
            grid_poly = Polygon([
                [x, y],
                [x + meters_to_deg, y],
                [x + meters_to_deg, y + meters_to_deg],
                [x, y + meters_to_deg],
                [x, y]
            ])
            if polygon.intersects(grid_poly):
                intersection = polygon.intersection(grid_poly)
                if not intersection.is_empty and intersection.is_valid:
                    grids.append(grid_poly)

    # Check if any grids were created
    if not grids:
        raise ValueError("No valid grid cells generated within the polygon")

    # Create GeoDataFrame with explicit geometry column and CRS
    gdf = gpd.GeoDataFrame(geometry=grids, crs="EPSG:4326")
    # Explicitly set the geometry column to ensure it's active
    gdf = gdf.set_geometry('geometry')
    if gdf.crs is None:
        raise ValueError("GeoDataFrame CRS is not set after initialization")
    if gdf.geometry.name != 'geometry':
        raise ValueError("Geometry column is not correctly set")

    # Compute centroids
    gdf['centroid'] = gdf.geometry.centroid
    gdf['grid_index'] = gdf.index
    return gdf

def assign_buildings_to_grids(buildings_geojson, grid_gdf):
    buildings_gdf = gpd.GeoDataFrame.from_features(buildings_geojson['features'], crs="EPSG:4326")

    # Set centroids correctly
    buildings_gdf['geometry'] = buildings_gdf.geometry.centroid
    buildings_gdf = buildings_gdf.set_geometry('geometry')

    # Perform spatial join WITHOUT renaming geometry
    joined = gpd.sjoin(buildings_gdf, grid_gdf, how='left', predicate='within')

    # Now count buildings per grid
    building_counts = joined['index_right'].value_counts().reindex(grid_gdf.index, fill_value=0)
    grid_gdf['building_count'] = building_counts

    # Assign grid indices to buildings
    buildings_geojson['features'] = [
        {**f, 'properties': {**f['properties'], 'grid_index': int(joined['index_right'].iloc[i]) if not pd.isna(
            joined['index_right'].iloc[i]) else -1}}
        for i, f in enumerate(buildings_geojson['features'])
    ]

    return grid_gdf, buildings_geojson

def grid_clustering(grid_centroids, building_counts, num_clusters, tolerance=0.1):
    """
    Cluster grids into num_clusters with building counts in [min_buildings, max_buildings].
    Splits alternate between vertical (x) and horizontal (- Splits are 1:1 (50%:50%) for even num_clusters, 1:2 (33.33%:66.67%) for odd num_clusters.

    Args:
        grid_centroids: np.array of shape (N, 2) with [x, y] coordinates
        building_counts: np.array of building counts per grid
        num_clusters: desired number of clusters (≥ 1)
        tolerance: allowable +/- tolerance in building counts per cluster

    Returns:
        np.array of cluster labels for each grid
    """
    n_grids = len(grid_centroids)
    if n_grids < num_clusters:
        raise ValueError(f"Cannot create {num_clusters} clusters with only {n_grids} grids")
    if num_clusters < 1:
        raise ValueError("Number of clusters must be at least 1")

    assigned = np.full(n_grids, fill_value=-1)  # -1 means unassigned
    total_buildings = building_counts.sum()
    target_buildings_per_cluster = total_buildings / num_clusters
    min_buildings = target_buildings_per_cluster * (1 - tolerance)
    max_buildings = target_buildings_per_cluster * (1 + tolerance)

    def recursive_split(indices, num_clusters_to_assign, cluster_id_start, is_vertical=True):
        """
        Recursively split the given grid indices into num_clusters_to_assign clusters.

        Args:
            indices: np.array of grid indices to cluster
            num_clusters_to_assign: number of clusters to create in this region
            cluster_id_start: starting cluster ID for assignments
            is_vertical: True for vertical split (x), False for horizontal (y)

        Returns:
            next available cluster ID
        """
        if num_clusters_to_assign == 0:
            return cluster_id_start
        if num_clusters_to_assign == 1:
            # Base case: assign all grids to one cluster
            region_buildings = building_counts[indices].sum()
            if region_buildings < min_buildings or region_buildings > max_buildings:
                print(
                    f"Warning: Cluster {cluster_id_start} has {region_buildings} buildings, outside [{min_buildings}, {max_buildings}]")
            assigned[indices] = cluster_id_start
            return cluster_id_start + 1

        # Determine split ratio
        if num_clusters_to_assign % 2 == 0:
            split_ratio = 0.5
            left_clusters = num_clusters_to_assign // 2
            right_clusters = num_clusters_to_assign - left_clusters
        else:
            left_clusters = math.ceil(num_clusters_to_assign / 2)
            right_clusters = num_clusters_to_assign - left_clusters
            split_ratio = left_clusters / num_clusters_to_assign

        # Sort by x (vertical) or y (horizontal)
        coord_idx = 0 if is_vertical else 1
        sorted_indices = indices[np.argsort(grid_centroids[indices, coord_idx])]
        sorted_counts = building_counts[sorted_indices]
        cumulative_counts = np.cumsum(sorted_counts)
        total = cumulative_counts[-1] if len(cumulative_counts) > 0 else 0

        # Adjust split point to respect min/max constraints
        target = split_ratio * total
        split_idx = np.argmin(np.abs(cumulative_counts - target))
        left_count = cumulative_counts[split_idx] if split_idx < len(cumulative_counts) else 0
        right_count = total - left_count

        # Check if split is feasible
        left_min = min_buildings * left_clusters
        right_min = min_buildings * right_clusters
        left_max = max_buildings * left_clusters
        right_max = max_buildings * right_clusters

        if left_count < left_min or right_count < right_min:
            # Try to adjust split point to meet minimum
            for i in range(len(cumulative_counts)):
                lc = cumulative_counts[i]
                rc = total - lc
                if lc >= left_min and rc >= right_min and lc <= left_max and rc <= right_max:
                    split_idx = i
                    left_count = lc
                    right_count = rc
                    break
            else:
                print(
                    f"Warning: Cannot split {len(indices)} grids into {left_clusters}+{right_clusters} clusters within constraints")

        # Split indices
        left_indices = sorted_indices[:split_idx + 1]
        right_indices = sorted_indices[split_idx + 1:]

        # Recursively split left and right regions, alternating direction
        next_id = recursive_split(left_indices, left_clusters, cluster_id_start, not is_vertical)
        next_id = recursive_split(right_indices, right_clusters, next_id, not is_vertical)

        return next_id

    def post_process_assignments():
        """Adjust cluster assignments to meet min/max constraints."""
        for cluster_id in range(num_clusters):
            cluster_indices = np.where(assigned == cluster_id)[0]
            cluster_count = building_counts[cluster_indices].sum()
            if cluster_count < min_buildings:
                # Find grids from other clusters to add
                other_clusters = [i for i in range(num_clusters) if i != cluster_id]
                for other_id in other_clusters:
                    other_indices = np.where(assigned == other_id)[0]
                    other_count = building_counts[other_indices].sum()
                    if other_count > min_buildings:
                        # Sort by distance to cluster centroid
                        cluster_centroid = grid_centroids[cluster_indices].mean(axis=0)
                        distances = np.linalg.norm(grid_centroids[other_indices] - cluster_centroid, axis=1)
                        sorted_other = other_indices[np.argsort(distances)]
                        for idx in sorted_other:
                            if cluster_count < min_buildings and other_count > min_buildings:
                                assigned[idx] = cluster_id
                                cluster_count += building_counts[idx]
                                other_count -= building_counts[idx]
                            else:
                                break
            elif cluster_count > max_buildings:
                # Move grids to other clusters
                other_clusters = [i for i in range(num_clusters) if i != cluster_id]
                for other_id in other_clusters:
                    other_indices = np.where(assigned == other_id)[0]
                    other_count = building_counts[other_indices].sum()
                    if other_count < max_buildings:
                        cluster_centroid = grid_centroids[other_indices].mean(axis=0) if len(other_indices) > 0 else \
                        grid_centroids[cluster_indices].mean(axis=0)
                        distances = np.linalg.norm(grid_centroids[cluster_indices] - cluster_centroid, axis=1)
                        sorted_cluster = cluster_indices[np.argsort(distances)]
                        for idx in sorted_cluster:
                            if cluster_count > max_buildings and other_count + building_counts[idx] <= max_buildings:
                                assigned[idx] = other_id
                                cluster_count -= building_counts[idx]
                                other_count += building_counts[idx]
                            else:
                                break

    # Start recursive splitting with all grids
    recursive_split(np.arange(n_grids), num_clusters, 0, is_vertical=False)

    # Post-process to enforce constraints
    post_process_assignments()
    return assigned

@app.route('/get_building_density_v2', methods=['POST'])
def get_building_density_v2():
    try:
        data = request.get_json()
        polygon_coords = data.get("polygon", [])
        num_clusters = int(data.get("noOfClusters", 3))
        tolerance = float(data.get("thresholdVal", 10)) / 100
        grid_size = int(data.get("gridLength", 50))
        buildings_area_in_meters = float(data.get("buildingsAreaInMeters", 0))
        buildings_confidence = float(data.get("buildingsConfidence", 0)) / 100
        if not polygon_coords:
            return jsonify({"error": "Invalid polygon coordinates"}), 400

        # Fetch buildings
        buildings_geojson = getBuildingsDataFromDB(polygon_coords, buildings_area_in_meters, buildings_confidence)
        if not buildings_geojson['features']:
            return jsonify({"message": "No buildings found within the polygon", "building_count": 0}), 404

        # Generate grid
        grid_gdf = generate_grid(polygon_coords, grid_size)

        # Assign buildings to grids
        grid_gdf, buildings_geojson = assign_buildings_to_grids(buildings_geojson, grid_gdf)

        # Cluster grids
        coords = np.array([[point.x, point.y] for point in grid_gdf.centroid])
        weights = grid_gdf['building_count'].values
        valid_mask = weights > 0

        if not valid_mask.any():
            return jsonify(
                {"message": "No grids with buildings", "building_count": len(buildings_geojson['features'])}), 404

        valid_coords = coords[valid_mask]
        valid_weights = weights[valid_mask]
        total_buildings = valid_weights.sum()
        _, min_size, max_size = _calculate_cluster_sizes(total_buildings, num_clusters, tolerance)

        labels = grid_clustering(valid_coords, valid_weights, num_clusters)

        grid_gdf['cluster_label'] = -1
        grid_gdf.loc[valid_mask, 'cluster_label'] = labels

        # Assign cluster labels to buildings
        grid_cluster_map = grid_gdf.set_index('grid_index')['cluster_label'].to_dict()
        for feature in buildings_geojson['features']:
            grid_index = feature['properties'].get('grid_index', -1)
            feature['properties']['cluster_label'] = grid_cluster_map.get(grid_index, -1)

        # Convert grids to GeoJSON, excluding the centroid column
        grid_gdf_for_json = grid_gdf.drop(columns=['centroid'])  # Drop non-serializable centroid column
        grid_geojson = json.loads(grid_gdf_for_json.to_json())

        # Prepare clusters data
        clusters = [
            {
                "coordinates": feature['properties']['longitude_latitude']['coordinates'],
                "cluster": feature['properties']['cluster_label'],
                "grid_index": feature['properties']['grid_index']
            }
            for feature in buildings_geojson['features']
            if feature['properties']['cluster_label'] != -1
        ]

        return jsonify({
            "building_count": len(buildings_geojson['features']),
            "buildings": buildings_geojson,
            "grids": grid_geojson,
            "clusters": clusters
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
