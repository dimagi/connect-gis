import json
from sqlalchemy import create_engine, Column, Integer, String, JSON, Float, text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.inspection import inspect
from geoalchemy2 import Geometry

DATABASE_URI = "postgresql://username:password@localhost:5432/your_database"

# Initialize SQLAlchemy
engine = create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()


# Define the Ward model
class Ward(Base):
    __tablename__ = 'wards'

    id = Column(Integer, primary_key=True)
    country_name = Column(String)
    country_code = Column(String)
    state_name = Column(String)
    state_code = Column(String)
    lga_name = Column(String)
    lga_code = Column(String)
    ward_name = Column(String)
    ward_code = Column(String)
    global_id = Column(String)
    source = Column(String)
    source_date = Column(String)
    properties = Column(JSON)  # Stores the nested properties
    population = Column(Float)  # population_1 from GeoJSON
    geom = Column(Geometry(geometry_type='MULTIPOLYGON', srid=4326))


# Create table if not exists
def create_table():
    inspector = inspect(engine)

    # Check if table exists
    if not inspector.has_table('wards'):
        # Enable PostGIS if not already enabled
        session.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        session.commit()

        # Create the table
        Base.metadata.create_all(engine)

        # Create indexes
        index_statements = [
            "CREATE INDEX idx_wards_geom ON wards USING GIST (geom)",
            "CREATE INDEX idx_wards_ward ON wards (ward_name)",
            "CREATE INDEX idx_wards_ward_code ON wards (ward_code)",
        ]

        for statement in index_statements:
            try:
                session.execute(text(statement))
                session.commit()
            except Exception as e:
                print(f"Error creating index: {e}")
                session.rollback()

        print("Table 'wards' and indexes created successfully")
    else:
        print("Table 'wards' already exists")


# Process and insert GeoJSON data
def insert_geojson_data(geojson_file):
    with open(geojson_file) as f:
        data = json.load(f)

    features = data['features']
    print(f"Found {len(features)} wards to insert")

    for feature in features:
        props = feature['properties']
        geometry = feature['geometry']

        # Convert MultiPolygon coordinates to WKT format
        coordinates = geometry['coordinates']
        wkt_coords = []
        for polygon in coordinates:
            polygon_coords = []
            for ring in polygon:
                ring_coords = [" ".join(map(str, coord)) for coord in ring]
                polygon_coords.append(f"({', '.join(ring_coords)})")
            wkt_coords.append(f"({', '.join(polygon_coords)})")
        wkt_geom = f"MULTIPOLYGON({', '.join(wkt_coords)})"

        ward = Ward(
            country_name=props.get('country_name'),
            country_code=props.get('country_code'),
            state_name=props.get('state_name'),
            state_code=props.get('state_code'),
            lga_name=props.get('lga_name'),
            lga_code=props.get('lga_code'),
            ward_name=props.get('ward_name'),
            ward_code=props.get('ward_code'),
            global_id=props.get('global_id'),
            source=props.get('source'),
            source_date=props.get('source_date'),
            properties=props.get('properties', {}),
            population=props.get('population_1'),
            geom=f"SRID=4326;{wkt_geom}"
        )

        session.add(ward)

    session.commit()
    print(f"Successfully inserted {len(features)} wards")


if __name__ == "__main__":
    GEOJSON_FILE = "../static/nigeria_ward.geojson"

    try:
        create_table()
        insert_geojson_data(GEOJSON_FILE)

    except Exception as e:
        print(f"Error: {e}")
        session.rollback()
    finally:
        session.close()