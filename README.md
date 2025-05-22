# Building Density and Clustering API

This is a Flask-based web application that provides APIs for fetching building data, performing clustering (K-means and grid-based), and generating reports based on geospatial data. The application integrates with Google Earth Engine (GEE) and a PostgreSQL database with PostGIS extension to process building data within specified polygons or around pin locations. It supports clustering buildings using balanced K-means or grid-based clustering and generates CSV reports summarizing ward-level visit data.

## Features
- **Building Data Retrieval**: Fetches building data within a polygon from either Google Earth Engine or a PostgreSQL database, with optional filters for minimum area and confidence.
- **Clustering**:
  - **K-means Clustering**: Performs balanced K-means clustering based on a specified number of clusters or buildings per cluster.
  - **Grid-based Clustering**: Generates a grid over a polygon, assigns buildings to grid cells, and clusters grids to balance building counts.
- **Reporting**: Generates CSV reports summarizing ward-level visit data, with options to include building-to-visit distance metrics.
- **Geospatial Support**: Uses PostGIS for spatial queries and GeoPandas for handling geospatial data.
- **CORS Support**: Allows cross-origin requests for frontend integration.

## Prerequisites
- **Python**: Version 3.8 or higher.
- **PostgreSQL**: With PostGIS extension enabled for spatial queries.
- **Google Earth Engine Account**: For accessing building data via GEE.
- **Dependencies**: Listed in `requirements.txt`.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Thushar12E45/dimagi-map-project.git
   cd dimagi-map-project
   ```

2. **Set Up a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   Create a `.env` file in the project root with the following variables:
   ```
   GEE_CREDS=<your-google-earth-engine-credentials-json>
   GEE_PROJECT_NAME=<your-google-earth-engine-project-name>
   DB_USER=<your-postgres-username>
   DB_PASSWORD=<your-postgres-password>
   DB_HOST=<your-postgres-host>
   DB_PORT=<your-postgres-port>
   DB_NAME=<your-postgres-database-name>
   HOST_URL=<your-host-url>  # Optional, defaults to https://connectgis.dimagi.com
   ```
   - `GEE_CREDS`: JSON string of Google Earth Engine service account credentials.
   - Database credentials for PostgreSQL connection.


5. **Database Setup**
   - Ensure the PostgreSQL database is running and has the PostGIS extension enabled.
   - The `buildings` table should contain building data with spatial geometry.


6. **Run the Application**
   ```bash
   python app.py
   ```
   The app runs on `http://0.0.0.0:5000` in debug mode by default.

# Google Earth Engine (GEE) Setup Guide

## Prerequisites
- Google Cloud account
- Earth Engine access (sign up at [earthengine.google.com](https://earthengine.google.com/))

## Setup Steps

### 1. Create a Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Create Project"
3. Enter project name and details
4. Click "Create"

### 2. Register Project for Earth Engine
1. After project creation, register it for:
   - **Commercial use** (if applicable), or
   - **Non-commercial use** (for research/academic purposes)
2. Wait for approval (typically 1-2 business days)

### 3. Enable Earth Engine API
1. Search / Navigate to: **APIs & Services** → **Library**
2. Search for "Google Earth Engine API"
3. Click "Enable"

### 4. Local Development Setup
```bash
# Install Earth Engine Python API
pip install earthengine-api

# Authenticate (will open browser)
earthengine authenticate
```

### 4. Configuration
Add your GEE project name to the .env file
```dotenv
GEE_PROJECT_NAME=<your-google-earth-engine-project-name>
```

## Important Notes
- **Local Development**: Authentication via **earthengine authenticate** is sufficient for local use (no credentials needed in **.env**)
- **Production / Other Envinorments**: ou must add the appropriate GEE credentials to your .env file:
```dotenv
GEE_CREDS=<your-service-account-credentials-json>
```
## API Endpoints

### 1. Home (`/`)
- **Method**: GET
- **Description**: Renders the `index.html` template with the configured `HOST_URL`.
- **Response**: HTML page.

### 2. Get Building Density (`/get_building_density`)
- **Method**: POST
- **Description**: Fetches building data within a polygon or around a pin and performs clustering.
- **Request Body**:
  ```json
  {
    "clusteringType": "kMeans|balancedKMeans|bottomUp",
    "noOfClusters": <int>,  // Number of clusters (default: 3)
    "noOfBuildings": <int>,  // Target buildings per cluster (default: 250)
    "buildingsAreaInMeters": <float>,  // Minimum building area (default: 0)
    "buildingsConfidence": <int>,  // Minimum confidence (0-100, default: 0)
    "thresholdVal": <int>,  // Tolerance percentage (default: 10)
    "fetchClusters": <boolean>,  // Whether to perform clustering (default: false)
    "dbType": "GEE|DB",  // Data source (Google Earth Engine or Database)
    "polygon": [[lng, lat], ...],  // Polygon coordinates (for kMeans/balancedKMeans)
    "pin": [lng, lat]  // Pin coordinates (for bottomUp)
  }
  ```
- **Response**: JSON with building count, GeoJSON features, and optional cluster data.
- **Example Response**:
  ```json
  {
    "building_count": 100,
    "buildings": {"type": "FeatureCollection", "features": [...]},
    "clusters": [{"coordinates": [lng, lat], "cluster": <int>, "numOfBuildings": <int>}, ...]
  }
  ```

### 3. Get Building Density V2 (`/get_building_density_v2`)
- **Method**: POST
- **Description**: Fetches buildings from the database, generates a grid, assigns buildings to grid cells, and performs grid-based clustering.
- **Request Body**:
  ```json
  {
    "polygon": [[lng, lat], ...],  // Polygon coordinates
    "noOfClusters": <int>,  // Number of clusters (default: 3)
    "thresholdVal": <int>,  // Tolerance percentage (default: 10)
    "gridLength": <int>,  // Grid size in meters (default: 50)
    "buildingsAreaInMeters": <float>,  // Minimum building area (default: 0)
    "buildingsConfidence": <int>  // Minimum confidence (0-100, default: 0)
  }
  ```
- **Response**: JSON with building count, GeoJSON features, grid GeoJSON, and cluster data.
- **Example Response**:
  ```json
  {
    "building_count": 100,
    "buildings": {"type": "FeatureCollection", "features": [...]},
    "grids": {"type": "FeatureCollection", "features": [...]},
    "clusters": [{"coordinates": [lng, lat], "cluster": <int>, "grid_index": <int>}, ...]
  }
  ```

### 4. Generate Report (`/generate_report`)
- **Method**: POST
- **Description**: Generates a CSV report summarizing ward-level visit data.
- **Request Body**:
  ```json
  {
    "data": [{"latitude": <float>, "longitude": <float>, "flw_id": <int>}, ...],
    "fetchVisitToBuildingsVal": <boolean>  // Include building-to-visit distance metrics (default: true)
  }
  ```
- **Response**: CSV file (`Ward_summary_report.csv`) with ward visit summary data.
- **Example CSV Headers** (with `fetchVisitToBuildingsVal=true`):
  ```csv
  state_name,lga_name,ward_name,population,total.visits,total.buildings,num.phc.serve.ward,median.visit.to.phc,max.visit.to.phc,median.building.to.phc,max.buildings.to.phc,unique.flws,coverage,percent.building.100.plus.to.visit,percent.building.200.plus.to.visit,percent.building.500.plus.to.visit,percent.building.10000.plus.to.visit
  ```

## Project Structure
```
├── app.py              # Main Flask application
├── .env                # Environment variables (not tracked)
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Frontend template
```

## Notes
- **Google Earth Engine**: Ensure valid GEE credentials are provided in `.env` for the `/get_building_density` endpoint with `dbType=GEE`.
- **Performance**: For large polygons, reduce the number of buildings or grid cells to avoid GEE’s 5000-element limit.
- **Security**: Sanitize inputs to prevent SQL injection (handled via parameterized queries in the code).
- **CORS**: Configured to allow all origins (`*`). Adjust in production for security.

## Troubleshooting
- **GEE Initialization Error**: Verify `GEE_CREDS` in `.env` and ensure the credentials file is correctly formatted.
- **Database Connection Error**: Check PostgreSQL credentials and ensure the database is accessible.
- **No Buildings Found**: Ensure the polygon or pin coordinates are valid and contain buildings in the database or GEE dataset.
- **Clustering Issues**: Adjust `thresholdVal` or reduce `noOfClusters`/`noOfBuildings` if clustering fails due to insufficient data.
