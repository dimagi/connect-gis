from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from geoalchemy2 import Geometry
from geoalchemy2.functions import ST_Within
from shapely.geometry import Polygon
from flask_cors import CORS
import json
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
import pandas as pd
import geopandas as gpd
from shapely.wkb import loads
import ee
import geemap
from collections import Counter
import torch
import datetime
from sklearn.metrics import pairwise_distances
import os
app = Flask(__name__)
CORS(app)

# Setup Flask SQLAlchemy and PostGIS connection
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:root@localhost:5432/open_builidings_data'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

try:
    # Ensure the directory exists
    os.makedirs(os.path.expanduser("~/.config/earthengine"), exist_ok=True)

    # Write credentials from Render environment variable
    credentials_path = os.path.expanduser("~/.config/earthengine/credentials")
    if not os.path.exists(credentials_path):
        with open(credentials_path, "w") as f:
            f.write(os.environ.get("EEDA_CREDENTIALS", "{}"))
    ee.Initialize(
        opt_url='https://earthengine-highvolume.googleapis.com'
    )
    buildings = ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons")
except Exception as e:
    print(f"Error initializing Earth Engine: {e}")


# Define your Building model with GeoAlchemy2
class Building(db.Model):
    __tablename__ = 'nigeria'

    id = db.Column(db.Integer, primary_key=True)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    area_in_meters = db.Column(db.Float)
    confidence = db.Column(db.Float)
    geometry = db.Column(Geometry('MULTIPOLYGON', srid=4326))  # PostGIS geometry column
    full_plus_code = db.Column(db.String)


def optimized_balanced_kmeans1(buildingsGDF, coords, num_clusters=3, max_iter=300, balance_tolerance=0.05):
    n_samples = len(coords)
    ideal_size = n_samples // num_clusters
    lower_bound = int(ideal_size * (1 - balance_tolerance))
    upper_bound = int(ideal_size * (1 + balance_tolerance))

    # Step 1: Initial KMeans Clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto', max_iter=max_iter)
    labels = kmeans.fit_predict(coords)

    # Step 2: Balancing without Overlaps
    cluster_counts = np.bincount(labels, minlength=num_clusters)

    for i in range(num_clusters):
        while cluster_counts[i] > upper_bound:
            # Find the furthest point in the over-full cluster
            cluster_i_indices = np.where(labels == i)[0]
            distances = pairwise_distances(coords[cluster_i_indices], [kmeans.cluster_centers_[i]])
            furthest_idx = np.argmax(distances)
            point_to_move_idx = cluster_i_indices[furthest_idx]

            # Find the closest under-full cluster
            distances_to_other_clusters = pairwise_distances([coords[point_to_move_idx]], kmeans.cluster_centers_)
            distances_to_other_clusters[0, i] = np.inf  # Exclude the current cluster
            closest_cluster = np.argmin(distances_to_other_clusters)

            # Check if moving the point maintains non-overlapping clusters
            if cluster_counts[closest_cluster] < upper_bound:
                # Move the point
                labels[point_to_move_idx] = closest_cluster
                cluster_counts[i] -= 1
                cluster_counts[closest_cluster] += 1

    return getClusters(buildingsGDF, coords, labels)

def balanced_kmeans(buildingsGDF, coords, num_clusters=3):
    n = len(coords)
    # Calculate the ideal size for each cluster
    ideal_size = n // num_clusters

    # Initialize KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)

    # Fit the model
    labels = kmeans.fit_predict(coords)

    # Count the number of points in each cluster
    counts = np.bincount(labels)

    # Check if clusters are balanced
    while not all(abs(count - ideal_size) <= 1 for count in counts):
        # If not balanced, adjust clusters
        for i in range(num_clusters):
            if counts[i] > ideal_size:
                # Move excess points to the nearest cluster
                excess_points = counts[i] - ideal_size
                indices_to_move = np.where(labels == i)[0][:excess_points]

                for index in indices_to_move:
                    # Find the nearest cluster that is under capacity
                    nearest_cluster = np.argmin(
                        [np.linalg.norm(coords[index] - kmeans.cluster_centers_[j]) for j in range(num_clusters) if
                         counts[j] < ideal_size])
                    labels[index] = nearest_cluster  # Move point to the nearest under-capacity cluster

        # Recalculate counts after moving points
        counts = np.bincount(labels)

    return getClusters(buildingsGDF, coords, labels)


def cluster_buildings_with_size(buildingsGDF, coords, max_buildings, thresholdVal = 1):
    distance_threshold =  np.ptp(coords, axis=0).max()*0.1

    cluster_counter = 1  # Start cluster ID from 1
    cluster_labels = np.zeros(len(coords), dtype=int)  # Store cluster labels

    def hierarchical_clustering(coords, threshold):
        """ Recursively clusters and assigns unique IDs. """
        nonlocal cluster_counter

        if len(coords) == 0:
            return np.array([])  # Return empty array for empty input

        # Perform hierarchical clustering
        Z = linkage(coords, method='centroid')
        labels = fcluster(Z, threshold, criterion='distance')

        # Count buildings per cluster
        unique_clusters, counts = np.unique(labels, return_counts=True)
        new_labels = np.zeros_like(labels)

        for cluster_label, count in zip(unique_clusters, counts):
            if count > max_buildings and thresholdVal < 1:
                # Recursively split large clusters with a reduced threshold
                new_threshold = threshold * thresholdVal  # Reduce threshold dynamically
                subset_coords = coords[labels == cluster_label, :]

                # Recursively cluster only the subset
                subset_cluster_labels = hierarchical_clustering(subset_coords, new_threshold)

                # Assign unique cluster IDs within the split
                new_labels[labels == cluster_label] = subset_cluster_labels
            else:
                # Assign a new unique cluster ID
                new_labels[labels == cluster_label] = cluster_counter
                cluster_counter += 1

        return new_labels

    # Run hierarchical clustering and update labels
    cluster_labels = hierarchical_clustering(coords, distance_threshold)

    return getClusters(buildingsGDF, coords, cluster_labels)

def getClusters(buildingsGDF, coords, cluster_labels):
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
    buildingsGDF["cluster_label"] = cluster_labels

    return clusters

def cluster_buildings_dbscan(buildingsGDF, coords, minSamples):
        epsilon = np.ptp(coords, axis=0).max() * 0.02 # 2% of the maximum range

        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=epsilon, min_samples=minSamples)
        clustering = dbscan.fit(coords)
        return getClusters(buildingsGDF, coords, clustering.labels_)

def cluster_buildings_ward_threshold(buildingsGDF, coords):
    distance_threshold =  np.ptp(coords, axis=0).max()*0.1
    # Perform Hierarchical Clustering with a distance threshold
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, linkage='ward').fit(coords)
    return getClusters(buildingsGDF, coords, clustering.labels_)


def cluster_buildings_ward_k(buildingsGDF, coords, num_clusters=3):
    # Perform Hierarchical Clustering
    clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward').fit(coords)
    return getClusters(buildingsGDF, coords, clustering.labels_)

def cluster_buildings_kMeans(buildingsGDF, coords, num_clusters=3):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(coords)
    return  getClusters(buildingsGDF, coords, cluster_labels)

def gdf_from_db(polygon_coords):
    """
    Creates a GeoDataFrame from a SQLAlchemy query result.

    Args:
        query: A SQLAlchemy query object that returns Building objects.

    Returns:
        A GeoDataFrame containing the data from the query.
    """

    # Query using SQLAlchemy with GeoAlchemy2
    buildings_in_polygon = Building.query.filter(
        ST_Within(Building.geometry, db.func.ST_GeomFromText(Polygon(polygon_coords).wkt, 4326))
    ).all()

    data = []
    for building in buildings_in_polygon:
        wkb_data = building.geometry.desc  # Get WKB data from WKBElement
        shapely_geom = loads(wkb_data)
        data.append({
            'id': building.id,
            'latitude': building.latitude,
            'longitude': building.longitude,
            'area_in_meters': building.area_in_meters,
            'confidence': building.confidence,
            'full_plus_code': building.full_plus_code,
            'geometry': shapely_geom  # Keep the geometry object
        })
    # Convert to Pandas DataFrame
    df = pd.DataFrame(data)

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")  # Specify CRS

    return gdf



@app.route('/get_building_density', methods=['POST'])
def get_building_density():
    data = request.get_json()
    polygon_coords = data.get("polygon", [])
    thresholdVal = float(data.get("thresholdVal"))
    clusteringType = data.get("clusteringType")
    noOfClusters = data.get("noOfClusters")
    noOfBuildings = data.get("noOfBuildings")
    if not polygon_coords:
        return jsonify({"error": "Invalid polygon coordinates"}), 400

    # # Create EE Polygon geometry
    # region = ee.Geometry.Polygon(polygon_coords)
    # # Filter buildings inside the polygon
    # filtered_buildings = buildings.filterBounds(region)

    # buildings_geojson = filtered_buildings.getInfo()
    # coordinates = [feature['properties']['longitude_latitude']['coordinates'] for feature in buildings_geojson['features']]

    buildingsGDF = gdf_from_db(polygon_coords)
    coordinates = buildingsGDF[['longitude', 'latitude']].values.tolist()

    if not coordinates:
        return jsonify({"message": "No buildings found within the polygon"}), 404
    coordinates = np.array(coordinates)
    buildingsCount = len(coordinates)
    kVal = int(buildingsCount/int(noOfBuildings))
    # Perform clustering (or any other processing) as needed
    if(clusteringType == 'kMeans'):
        clusters = cluster_buildings_kMeans(buildingsGDF, coordinates, int(noOfClusters))
    elif(clusteringType == 'greedyDivision'):
        clusters = cluster_buildings_kMeans(buildingsGDF, coordinates, int(kVal))
    elif(clusteringType == 'balancedKMeans'):
        clusters = optimized_balanced_kmeans1(buildingsGDF, coordinates, balance_tolerance=float(thresholdVal/100))
    elif (clusteringType == 'hierarchicalClustering'):
        clusters = cluster_buildings_with_size(buildingsGDF, coordinates, 100, thresholdVal)
    elif (clusteringType == 'dbScan'):
        clusters = cluster_buildings_dbscan(buildingsGDF, coordinates, int(noOfBuildings))

    return jsonify({
        "building_count": buildingsCount,
        "buildings": json.loads(buildingsGDF.to_json()),
        "clusters": clusters,
    })


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
